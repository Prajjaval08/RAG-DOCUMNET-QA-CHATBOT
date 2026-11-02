"""Regression evaluation: run an eval set against the live index.

Usage:

    python scripts/evaluate.py --eval-file eval/eval_set.example.jsonl --report eval/report.json

Each eval line is JSON with:
    {"question": "...", "expected_sources": ["file.txt"],
     "expected_keywords": ["word the answer should contain"]}

Retrieval-only mode (--no-llm) measures retrieval recall / hit rate and
latency without spending LLM calls. With an LLM configured (see .env),
it also measures answer keyword coverage. Keyword coverage is a
regression signal, not proof of correctness — review failing cases
by hand before changing the pipeline on its basis alone.

Exit code 0 always on a completed run (a low score is a report, not a
crash); exit code 2 only for configuration errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_chatbot.config import load_config
from rag_chatbot.embeddings import create_embedder
from rag_chatbot.evaluation import evaluate_cases
from rag_chatbot.llm import create_llm_provider
from rag_chatbot.rag import answer_question
from rag_chatbot.vector_store import FAISSVectorStore


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--no-llm", action="store_true", help="Retrieval metrics only")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        config = load_config()
        cases = [
            json.loads(line)
            for line in Path(args.eval_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not cases:
            raise ValueError(f"No eval cases found in {args.eval_file}")
        store = FAISSVectorStore.load(
            config.index_dir,
            config.embedding_provider,
            config.effective_embedding_model,
        )
        embedder = create_embedder(
            config.embedding_provider,
            config.effective_embedding_model,
            config.ollama_base_url,
        )
        top_k = args.top_k or config.retrieval_top_k
        llm = None
        if not args.no_llm:
            llm = create_llm_provider(
                config.llm_provider,
                config.llm_model,
                config.llm_temperature,
                config.ollama_base_url,
            )

        def retrieve(question: str) -> list[str]:
            if store.is_empty:
                return []
            vector = embedder.embed_query(question)
            return [r.chunk.filename for r in store.search(vector, top_k)]

        def answer(question: str) -> str:
            assert llm is not None
            return answer_question(question, store, embedder, llm, top_k).answer

        report = evaluate_cases(
            cases, retrieve, None if args.no_llm else answer
        )
    except Exception as exc:  # noqa: BLE001 - script reports config errors plainly
        print(f"evaluate failed: {exc}", file=sys.stderr)
        return 2

    payload = report.as_dict()
    # Keep the report readable: answers are long, keep them only per-case.
    print(
        json.dumps(
            {k: v for k, v in payload.items() if k != "cases"},
            indent=2,
        )
    )
    for case in payload["cases"]:
        print(
            f"- recall={case['recall']:.2f} hit={case['hit']} "
            f"keywords={case['keyword_coverage']:.2f} "
            f"{case['latency_ms']:.0f}ms :: {case['question']}"
        )
    if args.report:
        Path(args.report).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
