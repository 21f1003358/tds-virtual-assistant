from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import Optional
import pandas as pd
import faiss
import os
from dotenv import load_dotenv
import requests
import base64
from PIL import Image
import pytesseract
import io
from functools import lru_cache

# Load environment variables
load_dotenv()

# Set tesseract path from .env if not already in PATH
if not shutil.which("tesseract"):
    tesseract_path = os.getenv("TESSERACT_CMD")
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Base directory for locating static files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "..", "assets", "discourse_faiss_index.index")
CSV_PATH = os.path.join(BASE_DIR, "..", "assets", "discourse_faiss_texts.csv")

# Initialize app and model
app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

class Query(BaseModel):
    question: str
    k: Optional[int] = 5
    image: Optional[str] = None  # base64-encoded image

@lru_cache(maxsize=1)
def load_data():
    index = faiss.read_index(INDEX_PATH)
    df = pd.read_csv(CSV_PATH)
    texts = df["chunk_cleaned"].dropna().tolist()
    return index, texts

def retrieve_chunks(query, index, texts, model, k):
    question_embedding = model.encode([query])
    _, I = index.search(question_embedding, k)
    return [texts[i] for i in I[0] if i < len(texts)]

def generate_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)
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
def health_check():
    return {"message": "API is live!"}

@app.post("/api")
def ask(query: Query):
    # OCR from base64 image
    if query.image:
        try:
            image_data = base64.b64decode(query.image)
            image = Image.open(io.BytesIO(image_data))
            ocr_text = pytesseract.image_to_string(image)
            query.question += f"\n\n[Extracted from image: {ocr_text}]"
        except Exception as e:
            print("Image OCR failed:", e)

    index, texts = load_data()
    chunks = retrieve_chunks(query.question, index, texts, model, query.k or 5)
    answer = generate_answer(query.question, chunks)

    links = []
    for chunk in chunks:
        if "http" in chunk:
            part = chunk.split("http", 1)[1].split()[0].strip().rstrip(".,)\"")
            links.append({"url": "http" + part, "text": chunk[:150] + "..."})

    return {
        "answer": answer,
        "links": links[:2]  # Return at most 2 links
    }
