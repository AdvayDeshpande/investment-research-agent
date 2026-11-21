"""PDF ingestion pipeline: PDF -> text per page -> token chunks -> embeddings -> ChromaDB.

Usage:
    python ingestion/ingest.py data/documents/
    python ingestion/ingest.py data/documents/ --reset    # wipe the collection first

Idempotent: chunk IDs are derived from file name + content hash, and all existing chunks of a
file are replaced on re-ingestion, so re-running never creates duplicates.
"""
import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Allow running as a script (python ingestion/ingest.py) from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from tools.document_search import get_collection, get_embedding_model  # noqa: E402

logger = logging.getLogger("ingestion")


def extract_pages(pdf_path: Path) -> List[Dict]:
    """Return [{'page': 1, 'text': '...'}, ...] for all pages with extractable text."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.warning("Could not read page %d of %s: %s", number, pdf_path.name, exc)
            continue
        text = " ".join(text.split())
        if text:
            pages.append({"page": number, "text": text})
    return pages


def chunk_text(text: str, tokenizer, chunk_size: int = config.CHUNK_SIZE_TOKENS,
               overlap: int = config.CHUNK_OVERLAP_TOKENS) -> List[str]:
    """Split text into windows of `chunk_size` tokens that overlap by `overlap` tokens."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks, step = [], chunk_size - overlap
    for start in range(0, len(token_ids), step):
        window = token_ids[start:start + chunk_size]
        chunks.append(tokenizer.decode(window).strip())
        if start + chunk_size >= len(token_ids):
            break
    return [c for c in chunks if c]


def ingest_file(pdf_path: Path, collection, model) -> int:
    pages = extract_pages(pdf_path)
    if not pages:
        logger.warning("No extractable text in %s (scanned PDF?) - skipped", pdf_path.name)
        return 0

    ids, documents, metadatas = [], [], []
    for page in pages:
        for i, chunk in enumerate(chunk_text(page["text"], model.tokenizer)):
            digest = hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:12]
            ids.append(f"{pdf_path.name}-p{page['page']}-c{i}-{digest}")
            documents.append(chunk)
            metadatas.append({"source": pdf_path.name, "page": page["page"], "chunk": i})

    # Replace previous version of this file -> safe to re-run.
    collection.delete(where={"source": pdf_path.name})
    embeddings = model.encode(documents, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings.tolist())
    logger.info("%s: %d pages -> %d chunks stored", pdf_path.name, len(pages), len(ids))
    return len(ids)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    parser = argparse.ArgumentParser(description="Ingest PDFs into the local ChromaDB vector store")
    parser.add_argument("directory", nargs="?", default=str(config.DOCUMENTS_DIR))
    parser.add_argument("--reset", action="store_true", help="Delete the whole collection before ingesting")
    args = parser.parse_args()

    directory = Path(args.directory)
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        logger.error("No PDF files found in %s", directory.resolve())
        return 1

    if args.reset:
        import chromadb

        try:
            chromadb.PersistentClient(path=str(config.CHROMA_DIR)).delete_collection(config.CHROMA_COLLECTION)
            logger.info("Collection reset")
        except Exception:
            pass

    collection = get_collection(create=True)
    model = get_embedding_model()
    total = 0
    for pdf in pdfs:
        try:
            total += ingest_file(pdf, collection, model)
        except Exception:
            logger.exception("Failed to ingest %s", pdf.name)
    logger.info("Done. %d chunks from %d file(s). Collection now holds %d chunks at %s",
                total, len(pdfs), collection.count(), config.CHROMA_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
