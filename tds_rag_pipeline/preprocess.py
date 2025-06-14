from utils import extract_zip, load_markdown_texts, load_discourse_json, clean_html
from chunker import chunk_texts
from embedder import embed_and_store
import pandas as pd
import os

# Paths
md_zip = "tds_pages_md.zip"
discourse_zip = "discourse_json.zip"
md_extract = "tds_md"
discourse_extract = "discourse_json"

# Extract
extract_zip(md_zip, md_extract)
extract_zip(discourse_zip, discourse_extract)

# Load
md_texts = load_markdown_texts(os.path.join(md_extract, "tds_pages_md"))
df_discourse = load_discourse_json(os.path.join(discourse_extract, "discourse_json"))
df_discourse = df_discourse.dropna(subset=["cooked"])
discourse_texts = df_discourse["cooked"].astype(str).tolist()

# Combine and Chunk
all_texts = discourse_texts + md_texts
chunks = chunk_texts(all_texts)
df_chunks = pd.DataFrame({"chunk": chunks})
df_chunks["chunk_cleaned"] = df_chunks["chunk"].apply(clean_html)
df_chunks.to_csv("chunked_discourse_texts_cleaned.csv", index=False)

# Embed and store
embed_and_store(df_chunks["chunk_cleaned"].dropna().tolist(), "discourse_faiss_index.index", "discourse_faiss_texts.csv")

#print("Pipeline complete. Run `uvicorn api:app --reload` to start the API.")
