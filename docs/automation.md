# Automation: scheduled re-indexing and regression evaluation

This is the project's automation layer for the "MuseAI connectors"
idea in the spec: a small, scheduler-agnostic script that any
connector, cron job, or Task Scheduler task can invoke. No vendor
connector is required, and no connector is claimed to be connected
here — the scripts are the runnable part, the scheduler is yours.

## 1. Watch a directory and re-index only new documents

```bash
python scripts/watch_and_reindex.py --watch-dir ./incoming_docs
```

- Only files not already in the FAISS store (SHA-256 dedup) are
  indexed, so a nightly run is idempotent: no new files, no changes.
- One corrupt file is reported in the JSON output, not fatal to the
  whole run. Exit code 2 means a configuration error (missing watch
  directory, embedding-profile mismatch, unreadable index) — that is
  the case a scheduler should alert on.
- `--rebuild` clears the index first for a full re-index (required
  after changing the embedding provider/model).
- `--report report.json` writes the same JSON to a file for a
  connector to pick up and forward.

Schedule it nightly, for example with cron:

```cron
0 2 * * * cd /path/to/rag-document-qa-chatbot && python scripts/watch_and_reindex.py --watch-dir ./incoming_docs --report ./eval/last_watch_report.json
```

On Windows, create a Task Scheduler task that runs the same command
nightly. A Muse connector job, if you use one, only needs to run
this command and forward its stdout / report file on exit code 2.

Limitation: files deleted from the watch directory are NOT removed
from the index. Per-document delete is still on the README roadmap;
until then, use `--rebuild` after deletions.

## 2. Regression evaluation

```bash
python scripts/evaluate.py --eval-file eval/eval_set.example.jsonl --no-llm
python scripts/evaluate.py --eval-file eval/my_eval.jsonl --report eval/report.json
```

Each eval line is `{"question", "expected_sources", "expected_keywords"}`.
The report gives mean retrieval recall, hit rate, mean answer-keyword
coverage, and mean latency in ms. See `docs/evaluation.md` for what
each metric does and does not prove.

Run evaluation after any change to chunking, embeddings, retrieval,
or prompts, and keep the report next to the change — a falling recall
is the signal to stop and inspect the retrieved chunks before shipping.
