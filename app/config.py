import os
from dotenv import load_dotenv

load_dotenv()

# --- Gemini: both embeddings and LLM, one provider ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")

# --- Pinecone ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "legixo-takehome")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "docs")

# --- Ingest ---
DATA_DIR = os.getenv("DATA_DIR", "data")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# --- Retrieval / graph behaviour ---
TOP_K = int(os.getenv("TOP_K", "5"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.30"))
MAX_GRAPH_LOOPS = int(os.getenv("MAX_GRAPH_LOOPS", "2"))