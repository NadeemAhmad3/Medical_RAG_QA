import os
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
DATA_PATH = "dataset/mtsamples.csv"
VECTOR_DB_PATH = "vectorstore/faiss_index"

# Model Settings
EMBEDDING_MODEL = "embed-english-v3.0"  # Optimized for RAG
RERANK_MODEL = "rerank-english-v3.0"    # The "LLM Check" model
CHAT_MODEL = "command-r-plus-08-2024"   # Current Cohere chat model