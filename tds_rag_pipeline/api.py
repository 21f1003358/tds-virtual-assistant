from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import os
import requests
import base64
from PIL import Image
import pytesseract
import io
from dotenv import load_dotenv
from functools import lru_cache

from tds_rag_pipeline.embedder import retrieve_chunks

load_dotenv()

# Configure model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Google Drive raw export links
FAISS_URL = "https://drive.google.com/file/d/1lCypt1FVlcwzzMK1o3CiS0XhVb6cMTTC/view?usp=sharing"
CSV_URL = "https://drive.google.com/file/d/1WA4GDWPAesQ-mSXPwmDRm0nSC89DWyOr/view?usp=sharing"

def download_file(url: str, local_path: str):
    if not os.path.exists(local_path):
        print(f"Downloading {local_path}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.content)
            print(f"Saved {local_path}")
        else:
            raise RuntimeError(f"Download failed for {local_path} with status {response.status_code}")

@lru_cache(maxsize=1)
def load_index_and_texts():
    download_file(FAISS_URL, "discourse_faiss_index.index")
    download_file(CSV_URL, "discourse_faiss_texts.csv")

    index = faiss.read_index("discourse_faiss_index.index")
    df = pd.read_csv("discourse_faiss_texts.csv")
    df = df.tail(1000)
    texts = df["chunk_cleaned"].dropna().tolist()
    return index, texts

app = FastAPI()

class Query(BaseModel):
    question: str
    k: int = 5
    image: str | None = None

def generate_answer(question, chunks):
    context = "\n\n".join(chunks)
    prompt = f"""
You are a helpful assistant for a Data Science course.
Answer the question below using ONLY the following context.

Context:
{context}

Question: {question}
Answer:
"""
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "HTTP-Referer": "http://localhost",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

@app.get("/")
def read_root():
    return {"message": "API is live."}

@app.post("/api")
def ask(query: Query):
    if query.image:
        try:
            image_bytes = base64.b64decode(query.image)
            image = Image.open(io.BytesIO(image_bytes))
            ocr_text = pytesseract.image_to_string(image)
            query.question += f"\n\n[Extracted from image: {ocr_text}]"
        except Exception as e:
            print("Image OCR error:", e)

    index, texts = load_index_and_texts()
    chunks = retrieve_chunks(query.question, index, texts, model, query.k)
    answer = generate_answer(query.question, chunks)

    links = []
    for chunk in chunks:
        if "http" in chunk:
            parts = chunk.split("http", 1)
            url_candidate = "http" + parts[1].split()[0].strip().rstrip(",.)\"")
            links.append({
                "url": url_candidate,
                "text": chunk.strip()[:150] + "..."
            })

    return {
        "answer": answer,
        "links": links[:2]
    }
