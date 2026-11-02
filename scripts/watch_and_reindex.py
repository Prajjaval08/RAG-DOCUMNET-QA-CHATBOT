"""Scheduled re-index: scan a directory and index only new documents.

Usage (run nightly via cron / Task Scheduler / a Muse connector job):

    python scripts/watch_and_reindex.py --watch-dir ./incoming_docs

Exit code 0 even when individual files fail (errors are listed in the
JSON report); exit code 2 only for configuration errors (missing watch
directory, bad embedding profile, missing index). This lets a scheduler
alert on real failures without paging on one corrupt PDF.

The FAISS store is loaded with the embedding profile from the
environment (see .env.example); changing the embedding provider/model
requires clearing the index first, exactly as in the Streamlit UI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_chatbot.config import load_config
from rag_chatbot.embeddings import create_embedder
from rag_chatbot.vector_store import FAISSVectorStore
from rag_chatbot.watch import scan_directory


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-dir", required=True, help="Directory to scan")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Clear the existing index before scanning (full re-index)",
    )
    parser.add_argument("--report", default=None, help="Write JSON report to this path")
    args = parser.parse_args(argv)

    try:
        config = load_config()
        store = FAISSVectorStore.load(
            config.index_dir,
            config.embedding_provider,
            config.effective_embedding_model,
        )
        if args.rebuild:
            store.clear()
            store = FAISSVectorStore(
                config.index_dir,
                config.embedding_provider,
                config.effective_embedding_model,
            )
        embedder = create_embedder(
            config.embedding_provider,
            config.effective_embedding_model,
            config.ollama_base_url,
        )
        stats = scan_directory(
            args.watch_dir,
            store,
            embedder,
            config.chunk_size,
            config.chunk_overlap,
            config.max_upload_mb,
        )
    except Exception as exc:  # noqa: BLE001 - script reports config errors plainly
        print(f"watch_and_reindex failed: {exc}", file=sys.stderr)
        return 2

    report = {
        **stats.as_dict(),
        "indexed_chunks": len(store.chunks),
        "indexed_documents": len(store.doc_hashes),
        "note": (
            "Deleted files are not removed from the index; "
            "per-document delete is still on the roadmap."
        ),
    }
    print(json.dumps(report, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
