"""Configuration for the KYC compliance RAG demo."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "sample_docs"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
EVALS_FILE = BASE_DIR / "evals" / "golden_questions.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dimensional, runs locally on CPU
COLLECTION_NAME = "compliance_rules"

CHUNK_SIZE = 600       # characters; fits the embedding model's ~256-token window
CHUNK_OVERLAP = 100    # characters, so a sentence split across a boundary survives whole
TOP_K = 5
