# Evaluation: what is measured, and what it does not prove

Success criteria from the spec, and how this project measures each:

| Criterion | Metric in `src/rag_chatbot/evaluation.py` | Honest limit |
|---|---|---|
| Retrieval recall | `retrieval_recall`: expected source files retrieved / expected, averaged over cases | Measures file-level recall, not chunk-level relevance. A retrieved file can still contain the wrong passage. |
| Retrieval hit rate | `retrieval_hit`: at least one expected source retrieved | A weak bar — use it with recall, not instead of it. |
| Answer correctness | `answer_keyword_coverage`: expected keywords present in the answer | A regression signal only. Keyword presence is not correctness, and a correct paraphrase can miss a keyword. Review failing cases by hand. |
| Query latency | `latency_ms` per case (retrieve + generate) | Wall-clock on your machine with your models; not comparable across hardware or providers. |
| Low hallucination | Grounded prompt + fixed insufficient-information answer in `rag.py` | No automatic metric here proves non-hallucination. The prompt instructs grounding and the UI shows source excerpts — inspect them. |

## Writing a good eval set

- One case per line in JSONL: `question`, `expected_sources`
  (filenames as indexed, e.g. `"report.pdf"` or a URL for URL-indexed
  pages), `expected_keywords` (lowercase, distinctive words).
- Cover each document type you actually index, and include at least
  one question whose answer is NOT in the documents — the expected
  behaviour there is the fixed insufficient-information answer, not
  a guessed one.
- Keep the set small and stable; a 20-case set you actually run
  beats a 200-case set nobody maintains.

## Running it

See `docs/automation.md` section 2. Retrieval-only runs
(`--no-llm`) are free and fast; full runs need a configured LLM
provider and will spend API calls or local generation time.
