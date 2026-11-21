"""Central configuration. All values can be overridden via environment variables / .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# --- LLM (Groq) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# llama-3.3-70b-versatile is free on Groq and noticeably more reliable at tool calling
# than the 8B model. Set GROQ_MODEL=llama-3.1-8b-instant for higher free-tier rate limits.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "6"))

# --- External data APIs ---
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
NEWS_API_URL = "https://newsapi.org/v2/everything"
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "10"))
STOCK_DATA_TIMEOUT_SECONDS = float(os.getenv("STOCK_DATA_TIMEOUT_SECONDS", "15"))

# --- Local RAG ---
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "data" / "chroma_db")))
CHROMA_COLLECTION = "investment_documents"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
