# Implementation plan

Built locally in phases, following `docs/MUSE_AI_MASTER_PROMPT.md`:

1. **Inspect and plan** — read the master spec and the supplied README /
   GitHub-setup docs; define the module layout and validation rules.
2. **Minimal end-to-end slice** — TXT load → chunk → stub embed → FAISS →
   retrieve → stub LLM → cited answer (covered by `tests/test_smoke.py`).
3. **Expand capabilities** — PDF/DOCX/CSV loaders, SHA-256 dedup, persisted
   FAISS + manifest, embedding-profile guard, all four LLM providers,
   Streamlit sidebar (provider, model, temperature, top-k, embedding
   settings, KB status, safe clear-index), conversation history.
4. **Validate** — clean venv, `requirements-test.txt`, `pytest -q`,
   `ruff check .`, import checks, smoke test. Fix failures before finishing.
5. **Documentation and hardening** — README + docs match the real code;
   `.env.example` placeholders only; `.gitignore` verified; no secrets.
6. **Delivery (local only, per instruction)** — no GitHub repo was created
   and nothing was pushed. To publish later, follow `docs/GITHUB_SETUP.md`
   manually; do not claim the project is online until a push succeeds.

## Assumptions

- Python 3.11+; tests run on 3.12 in local validation.
- Local Sentence Transformers is the default embedding backend; it downloads
  weights on first real use (tests use a stub, so they need no weights).
- Live LLM smoke tests were not run: no API key or Ollama model was
  available in the build environment (see Limitations in the final report).
