# Troubleshooting

## "No extractable text" for a PDF
The PDF is likely scanned (image-only). OCR is not included, so scanned
PDFs cannot be indexed. Use a text-based PDF or run OCR externally first.

## Embedding-profile mismatch on load
You changed the embedding provider/model after indexing. This is by design:
vectors from a different embedding space must not be queried. Tick the
clear-index confirmation in the sidebar, click **Clear index**, then
re-index your documents.

## "OPENAI_API_KEY is not set" / similar
The selected provider needs its key in the environment (`.env`). Either set
it or switch the sidebar provider to Ollama (local, no key).

## Ollama connection error
Check Ollama is running (`ollama list`) and the model is pulled
(`ollama pull llama3.2`), and that `OLLAMA_BASE_URL` is reachable from
where the app runs (in Docker, host `localhost` usually does not work).

## FAISS import error
Install the test/runtime set: `pip install -r requirements-test.txt`
(for tests) or `pip install -r requirements.txt` (for the full app).

## Slow first question
The local Sentence Transformers model downloads on first use and loads
into memory once per process; later questions reuse the cached model.

## Answers say documents do not provide enough information
That is the grounding fallback working as designed: the retrieved chunks
did not support an answer. Rephrase, index more relevant documents, or
raise top-k in the sidebar — then inspect the cited excerpts.
