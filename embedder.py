import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def embed_and_store(texts, index_file, csv_file):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, index_file)
    pd.DataFrame({"chunk_cleaned": texts}).to_csv(csv_file, index=False)


def retrieve_chunks(query, index, texts, model=None, k=5):
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embed = model.encode([query])
    _, I = index.search(np.array(query_embed).astype("float32"), k)
    return [texts[i] for i in I[0]]
