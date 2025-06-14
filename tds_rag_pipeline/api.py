from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import os
from dotenv import load_dotenv
import base64
from PIL import Image
import io
import pytesseract
import shutil
from embedder import retrieve_chunks  # make sure it's accessible
import requests
load_dotenv()

# Optional: Set tesseract path from .env if not already in PATH
if not shutil.which("tesseract"):
    pytesseract_path = os.getenv("TESSERACT_CMD")
    if pytesseract_path:
        pytesseract.pytesseract.tesseract_cmd = pytesseract_path

app = FastAPI()

# Load FAISS index and chunk data
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

@app.post("/api")  # updated endpoint from /ask to /api
def ask(query: Query):
    # Extract text from base64 image if provided
    if query.image:
        try:
            image_bytes = base64.b64decode(query.image)
            image = Image.open(io.BytesIO(image_bytes))
            image.save("temp_image.png")
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
        "links": links[:2]  # limit to top 2 links
    }
