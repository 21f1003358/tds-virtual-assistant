import uvicorn

if __name__ == "__main__":
    uvicorn.run("tds_rag_pipeline.api:app", host="0.0.0.0", port=10000)