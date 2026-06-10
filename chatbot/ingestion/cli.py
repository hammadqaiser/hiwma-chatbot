"""CLI — Command-line tool to ingest PDF books into the vector store.

Usage:
    python -m chatbot.ingestion.cli --source ./Books
    python -m chatbot.ingestion.cli --source ./Books --config ./config.yaml
    python -m chatbot.ingestion.cli --stats
    python -m chatbot.ingestion.cli --clear
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from chatbot.config import load_config
from chatbot.ingestion.chunker import chunk_documents
from chatbot.ingestion.embedder import Embedder
from chatbot.ingestion.parser import parse_directory
from chatbot.ingestion.store import VectorStore


def ingest(
    source_dir: str,
    config_path: str | None = None,
    verbose: bool = True,
) -> dict:
    """Run the full ingestion pipeline: parse → chunk → embed → store.

    Args:
        source_dir: Path to directory containing PDF files.
        config_path: Optional path to config.yaml.
        verbose: Print progress messages.

    Returns:
        Dict with ingestion statistics.
    """
    config = load_config(config_path)
    start_time = time.time()

    # Step 1: Parse PDFs
    if verbose:
        print(f"\n📚 Parsing PDFs from: {source_dir}")
    documents = parse_directory(
        directory=source_dir,
        supported_formats=config.documents.supported_formats,
        exclude_patterns=config.documents.exclude_patterns,
    )
    if verbose:
        print(f"   Found {len(documents)} documents:")
        for doc in documents:
            print(f"   • {doc.title} ({doc.total_pages} pages)")

    if not documents:
        print("⚠️  No documents found. Check the source directory.")
        return {"documents": 0, "chunks": 0, "status": "empty"}

    # Step 2: Chunk documents
    if verbose:
        print(f"\n✂️  Chunking with size={config.ingestion.chunk_size}, overlap={config.ingestion.chunk_overlap}")
    chunks = chunk_documents(
        documents,
        chunk_size=config.ingestion.chunk_size,
        chunk_overlap=config.ingestion.chunk_overlap,
        min_chunk_size=config.ingestion.min_chunk_size,
    )
    if verbose:
        print(f"   Created {len(chunks)} chunks")

    # Step 3: Generate embeddings
    if verbose:
        print(f"\n🔢 Embedding with model: {config.retriever.embedding_model}")
        print("   (First run downloads the model — this may take a minute)")
    embedder = Embedder(model_name=config.retriever.embedding_model)
    embedded_chunks = embedder.embed_chunks(chunks, batch_size=32)
    if verbose:
        print(f"   Embedded {len(embedded_chunks)} chunks ({embedder.dimension}D vectors)")

    # Step 4: Store in vector database
    if verbose:
        print(f"\n💾 Storing in ChromaDB: {config.retriever.persist_directory}")
    store = VectorStore(
        collection_name=config.retriever.collection_name,
        persist_directory=config.retriever.persist_directory,
    )
    added = store.add_chunks(embedded_chunks)

    elapsed = time.time() - start_time

    stats = {
        "documents": len(documents),
        "total_pages": sum(d.total_pages for d in documents),
        "chunks": len(chunks),
        "embedded": len(embedded_chunks),
        "stored": added,
        "dimension": embedder.dimension,
        "elapsed_seconds": round(elapsed, 1),
        "status": "success",
    }

    if verbose:
        print(f"\n✅ Ingestion complete in {stats['elapsed_seconds']}s!")
        print(f"   Documents: {stats['documents']}")
        print(f"   Pages: {stats['total_pages']}")
        print(f"   Chunks: {stats['chunks']}")
        print(f"   Stored: {stats['stored']}")
        print(f"   Total in DB: {store.count()}")

    return stats


def show_stats(config_path: str | None = None) -> None:
    """Show vector store statistics."""
    config = load_config(config_path)
    store = VectorStore(
        collection_name=config.retriever.collection_name,
        persist_directory=config.retriever.persist_directory,
    )
    stats = store.get_stats()
    print(f"\n📊 Vector Store Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")


def clear_store(config_path: str | None = None) -> None:
    """Clear all data from the vector store."""
    config = load_config(config_path)
    store = VectorStore(
        collection_name=config.retriever.collection_name,
        persist_directory=config.retriever.persist_directory,
    )
    count = store.count()
    store.clear()
    print(f"\n🗑️  Cleared {count} chunks from '{config.retriever.collection_name}'")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest PDF books into the chatbot vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to directory containing PDF books (default: from config.yaml)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: ./config.yaml)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show vector store statistics",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all data from the vector store",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    if args.stats:
        show_stats(args.config)
        return

    if args.clear:
        clear_store(args.config)
        return

    # Determine source directory
    if args.source:
        source_dir = args.source
    else:
        config = load_config(args.config)
        source_dir = config.documents.source_dir

    if not Path(source_dir).exists():
        print(f"❌ Source directory not found: {source_dir}")
        sys.exit(1)

    ingest(source_dir, config_path=args.config, verbose=not args.quiet)


if __name__ == "__main__":
    main()
