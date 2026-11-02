# Step-by-step setup

## 1. Create a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

## 2. Configure one provider

Edit `.env` (never commit it). Easiest no-hosted-key path: install Ollama,
run `ollama pull llama3.2`, keep `LLM_PROVIDER=ollama`. For OpenAI /
Anthropic / Hugging Face, set only that provider's key variable.

## 3. Run

```bash
streamlit run app.py
```

Open the printed URL (usually `http://localhost:8501`), upload documents,
click **Index documents**, then ask a question.

## 4. Verify

```bash
pip install -r requirements-test.txt
pytest -q
ruff check .
```

## 5. Docker (optional)

```bash
docker compose up --build
```

Open `http://localhost:8501`. `data/` is mounted for persistence. If Ollama
runs on the host, `localhost` inside the container is not the host — set a
host-reachable `OLLAMA_BASE_URL`.
