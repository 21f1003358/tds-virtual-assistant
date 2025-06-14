import uvicorn
from tds_rag_pipeline.api import app  # <-- This imports `app` from your api.py

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)