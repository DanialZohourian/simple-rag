from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
DB_PATH = DATA_DIR / "rag.db"

MAX_UPLOAD_MB = 200
MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
OPENROUTER_X_TITLE = os.getenv("OPENROUTER_X_TITLE", "").strip()

EMBEDDING_MODEL = "openai/text-embedding-3-large"
CHAT_MODEL = "openai/gpt-4o"

TARGET_CHUNK_TOKENS = 2000
OVERLAP_TOKENS = 200
TOP_K = 6
