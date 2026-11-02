# Maintenance

- Before any release: run `pytest -q` and `ruff check .` in a clean venv.
- When behaviour or config changes, update README + the affected docs page
  in the same change; keep `docs/MUSE_AI_MASTER_PROMPT.md` in sync with the
  spec it came from.
- Check `git status` before every commit: `.env`, `data/faiss_index/`,
  uploads, `.venv/`, and model caches must never be staged (see `.gitignore`).
- Dependency bumps: prefer maintained packages; after changing provider
  SDK usage, re-check that provider's current docs (model ids and API
  shapes change over time — e.g. Anthropic model aliases).
- Known limitations to revisit: no OCR, no per-document delete, no auth /
  multi-user isolation, no streaming answers, no reranking. See README
  Roadmap.
