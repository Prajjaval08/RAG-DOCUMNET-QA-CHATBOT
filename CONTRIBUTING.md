# Contributing

1. Create a virtual environment and install dev dependencies:
   `pip install -r requirements-dev.txt`.
2. Run `pytest -q` and `ruff check .` before opening a pull request.
3. Tests must not call paid APIs or require model weights — use stub
   embedders/LLMs as in `tests/conftest.py`.
4. Never commit `.env`, real API keys, uploaded documents, or anything
   under `data/faiss_index/`. Check `git status` before staging.
5. Update the README and the relevant `docs/` page when behaviour or
   configuration changes.
