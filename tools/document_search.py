"""Local RAG tool: semantic search over ingested PDFs stored in ChromaDB."""
import logging
from functools import lru_cache
from typing import Any, Dict

import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the sentence-transformers model once and reuse it (shared with ingestion)."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model '%s'", config.EMBEDDING_MODEL)
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    # MiniLM is BERT-based (512 position embeddings); default is 256. Raise it so
    # 500-token chunks are embedded in full instead of being silently truncated.
    model.max_seq_length = 512
    return model


def get_collection(create: bool = False):
    """Return the ChromaDB collection (optionally creating it)."""
    import chromadb

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if create:
        return client.get_or_create_collection(config.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
    return client.get_collection(config.CHROMA_COLLECTION)


def search_documents(query: str, n_results: int = 4) -> Dict[str, Any]:
    """Return the most relevant text chunks from the local document store."""
    logger.info("search_documents(query=%r, n_results=%r)", query, n_results)
    if not query or not isinstance(query, str):
        return {"success": False, "error": "A search query is required."}
    try:
        n_results = max(1, min(int(n_results), 10))
    except (TypeError, ValueError):
        n_results = 4

    try:
        collection = get_collection()
    except Exception:
        return {"success": False,
                "error": "Document store not found. Run 'python ingestion/ingest.py data/documents/' first."}

    try:
        if collection.count() == 0:
            return {"success": False, "error": "Document store is empty. Add PDFs and run the ingestion script."}
        embedding = get_embedding_model().encode([query], normalize_embeddings=True).tolist()
        results = collection.query(query_embeddings=embedding, n_results=n_results)
    except Exception as exc:
        logger.exception("Document search failed")
        return {"success": False, "error": f"Document search failed: {exc}"}

    chunks = [
        {
            "text": doc,
            "source": meta.get("source"),
            "page": meta.get("page"),
            "relevance": round(1 - dist, 3),  # cosine similarity
        }
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ]
    return {"success": True, "query": query, "results": chunks}


SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Semantic search over the internal document library (annual reports, fund prospectuses, "
                       "research PDFs). Returns relevant text passages with source file and page number.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, phrased as a question or keywords"},
                "n_results": {"type": "integer", "description": "Number of passages to return (1-10). Default 4."},
            },
            "required": ["query"],
        },
    },
}
