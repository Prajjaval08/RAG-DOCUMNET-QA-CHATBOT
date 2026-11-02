"""Offline evaluation for the RAG pipeline.

Metrics implemented (from the project success criteria):

- ``retrieval_recall`` — fraction of expected source files returned in
  the top-k results. 1.0 means every expected document was retrieved.
- ``retrieval_hit`` — 1 if at least one expected source was retrieved.
- ``answer_keyword_coverage`` — fraction of expected answer keywords
  present in the generated answer (case-insensitive). This is a cheap
  regression signal, NOT a substitute for human review of correctness:
  an answer can contain the keywords and still be wrong, and a correct
  paraphrase can miss a keyword.
- ``latency_ms`` — wall-clock time for retrieve + generate per case.

A case is a dict with ``question`` (required), ``expected_sources``
(list of filenames that should be retrieved, optional) and
``expected_keywords`` (list of strings the answer should contain,
optional). ``evaluate_cases`` takes a ``retrieve`` callable
``question -> list of source filenames`` and an optional ``answer``
callable ``question -> answer text`` so the suite can run with a stub
LLM in tests and with the real pipeline via ``scripts/evaluate.py``.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass


def retrieval_recall(retrieved: list[str], expected: list[str]) -> float:
    """Recall@k over source filenames; 1.0 when there is nothing expected."""
    if not expected:
        return 1.0
    found = set(retrieved) & set(expected)
    return len(found) / len(expected)


def retrieval_hit(retrieved: list[str], expected: list[str]) -> bool:
    if not expected:
        return True
    return bool(set(retrieved) & set(expected))


def answer_keyword_coverage(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    lowered = answer.lower()
    found = sum(1 for kw in keywords if kw.lower() in lowered)
    return found / len(keywords)


@dataclass
class CaseResult:
    question: str
    retrieved_sources: list[str]
    recall: float
    hit: bool
    answer: str | None
    keyword_coverage: float
    latency_ms: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationReport:
    cases: list[CaseResult]
    mean_recall: float
    hit_rate: float
    mean_keyword_coverage: float
    mean_latency_ms: float

    def as_dict(self) -> dict:
        return {
            "mean_recall": self.mean_recall,
            "hit_rate": self.hit_rate,
            "mean_keyword_coverage": self.mean_keyword_coverage,
            "mean_latency_ms": self.mean_latency_ms,
            "cases": [c.as_dict() for c in self.cases],
        }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_cases(cases, retrieve, answer=None) -> EvaluationReport:
    """Run cases through retrieve (+ optional answer) and aggregate metrics."""
    results: list[CaseResult] = []
    for case in cases:
        question = case["question"]
        expected_sources = list(case.get("expected_sources", []))
        keywords = list(case.get("expected_keywords", []))
        started = time.perf_counter()
        retrieved = list(retrieve(question))
        answer_text = answer(question) if answer is not None else None
        latency_ms = (time.perf_counter() - started) * 1000.0
        results.append(
            CaseResult(
                question=question,
                retrieved_sources=retrieved,
                recall=retrieval_recall(retrieved, expected_sources),
                hit=retrieval_hit(retrieved, expected_sources),
                answer=answer_text,
                keyword_coverage=(
                    answer_keyword_coverage(answer_text, keywords)
                    if answer_text is not None
                    else 1.0
                ),
                latency_ms=latency_ms,
            )
        )
    return EvaluationReport(
        cases=results,
        mean_recall=_mean([c.recall for c in results]),
        hit_rate=_mean([1.0 if c.hit else 0.0 for c in results]),
        mean_keyword_coverage=_mean([c.keyword_coverage for c in results]),
        mean_latency_ms=_mean([c.latency_ms for c in results]),
    )
