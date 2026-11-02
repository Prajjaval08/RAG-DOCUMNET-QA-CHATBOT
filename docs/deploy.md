# Deploying live

Yes — this is a normal Streamlit app, so it can go live. The one
prerequisite is getting the code onto GitHub first (the zip in
`~/workspace/your_files/rag-document-qa-chatbot.zip` is still the
manual-push fallback while the Muse GitHub App push is blocked).
After that, pick one route. Both are free to start.

## Route A — Streamlit Community Cloud (easiest)

1. Push the project to a public GitHub repo.
2. Go to Streamlit Community Cloud → New app → pick the repo,
   branch `main`, main file `app.py` → Deploy.
3. In the app's dashboard → Settings → Secrets, add only what you use.
   The no-paid-key setup is Gemini (free AI Studio key) + local
   embeddings (no second key needed):
   ```toml
   LLM_PROVIDER = "gemini"
   LLM_MODEL = "gemini-3.8-flash"
   GEMINI_API_KEY = "your-free-google-ai-studio-key-here"
   EMBEDDING_PROVIDER = "local"
   ```
   Get the key at Google AI Studio (aistudio.google.com → Get API
   key → Create API key; free tier, no credit card, rate-limited).
   If you later get an OpenAI key, swap to `LLM_PROVIDER = "openai"`,
   `LLM_MODEL = "gpt-4o-mini"`, `OPENAI_API_KEY = "..."` instead.
   `app.py` copies these secrets into the environment on startup
   (Streamlit secrets are not environment variables by themselves),
   so no code change is needed for this route.
4. Open the Cloud URL, upload a document, index it, ask a question.

## Route B — Render (Docker-free, via render.yaml)

1. Push the project to GitHub.
2. Render dashboard → New → Web Service → connect the repo. Render
   picks up `render.yaml` (Python env, Streamlit start command,
   health check on `/_stcore/health`).
3. In Render → Environment, add the secret for your provider, e.g.
   `OPENAI_API_KEY`. The non-secret defaults in `render.yaml` already
   set OpenAI + local embeddings; change `LLM_PROVIDER` /
   `LLM_MODEL` there or in the dashboard to switch providers.
4. Deploy and open the Render URL.

## What changes when it goes live (read before sharing the link)

- **Ollama will not work hosted.** `localhost:11434` on Render /
  Streamlit Cloud is not your laptop. Use a hosted provider
  (Gemini with a free AI Studio key, OpenAI, Anthropic, or Hugging
  Face) for generation. Local
  Sentence Transformers embeddings still work, but the model
  downloads on first use, so the first indexing run is slow.
- **The index does not persist reliably on free tiers.** Render's
  free disk is ephemeral and Streamlit Cloud restarts wipe local
  files, so visitors must re-upload and re-index documents after a
  restart or redeploy. For a persistent knowledge base you need a
  paid persistent disk (Render) or an external vector store — that
  is a separate change, not in this project yet.
- **No login, no per-user isolation.** Anyone with the link can
  ask questions and spend your LLM API quota, and uploaded
  documents land in the shared index. Do not share the link
  publicly with a real API key attached, and do not upload private
  documents to a shared live demo.
- **Free tiers sleep.** Render's free service spins down when idle
  (first visit after idle takes ~a minute); Streamlit Cloud apps
  also sleep after inactivity.

## What is already in the project for this

- `Dockerfile` + `docker-compose.yml` for a self-hosted / VPS route:
  `docker compose up --build`, open port 8501.
- `render.yaml` for Route B.
- The Streamlit secrets bridge in `app.py` for Route A.
- `.env` is git-ignored — never commit a real key; use the
  platform's secrets / environment dashboard instead.
