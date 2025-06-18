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


def retrieve_chunks(query, index, texts, model, k):
    query_vector = model.encode([query])
    D, I = index.search(query_vector, k)

    # Prevent IndexError by filtering invalid indices
    if I is None or len(I) == 0 or len(I[0]) == 0:
        return []

    valid_chunks = [texts[i] for i in I[0] if 0 <= i < len(texts)]
    return valid_chunks
