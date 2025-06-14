from fastapi import FastAPI
from pydantic import BaseModel
from tds_rag_pipeline.embedder import retrieve_chunks
from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import os
import requests
from dotenv import load_dotenv
import base64
from PIL import Image
import io
import pytesseract

load_dotenv()

app = FastAPI()

# Load model and vector index
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("discourse_faiss_index.index")
df = pd.read_csv("discourse_faiss_texts.csv")
texts = df["chunk_cleaned"].dropna().tolist()

class Query(BaseModel):
    question: str
    k: int = 5
    image: str | None = None  # Optional base64 image input

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
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

@app.post("/ask")
def ask(query: Query):
    # If image is provided, extract text via OCR
    if query.image:
        try:
            image_bytes = base64.b64decode(query.image)
            image = Image.open(io.BytesIO(image_bytes))
            ocr_text = pytesseract.image_to_string(image)
            query.question += f"\n\n[Extracted from image: {ocr_text}]"
        except Exception as e:
            print("OCR failed:", e)

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
        "links": links[:5]  # at most 2 links
    }
