"""Vector store: ChromaDB, all-MiniLM-L6-v2 embeddings, HNSW + cosine similarity.

ChromaDB runs embedded in the Python process and persists to disk, so the corpus is
ingested once and reused across runs. Relevance is ``1.0 - cosine_distance``: 1.0 means
identical, 0.0 means unrelated.
"""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
)
from src.ingest import load_documents


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_fn,
    )


def ensure_ingested() -> int:
    """Ingest the corpus if the collection is empty. Returns the chunk count."""
    collection = get_collection()
    if collection.count() > 0:
        return collection.count()
    records = load_documents()
    collection.add(
        ids=[f"{r['source']}-{r['chunk_index']}" for r in records],
        documents=[r["text"] for r in records],
        metadatas=[
            {
                "source": r["source"],
                "rule_category": r["rule_category"],
                "chunk_index": r["chunk_index"],
            }
            for r in records
        ],
    )
    return len(records)


def query_rules(question: str, n_results: int = TOP_K) -> list[dict]:
    """Return the top-n chunks for a question, with text, metadata, and score."""
    ensure_ingested()
    collection = get_collection()
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    hits: list[dict] = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append(
            {
                "text": doc,
                "source": meta["source"],
                "rule_category": meta["rule_category"],
                "relevance_score": round(1.0 - float(dist), 4),
            }
        )
    return hits
