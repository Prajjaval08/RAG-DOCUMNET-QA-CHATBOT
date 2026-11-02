# Architecture

```
Upload (PDF/DOCX/TXT/MD/CSV, in memory)
  -> loaders.py      validate extension/size, extract text, keep PDF page
  -> chunking.py     RecursiveCharacterTextSplitter (or deterministic fallback)
  -> embeddings.py   local Sentence Transformers (default) or OpenAI
  -> vector_store.py FAISS IndexFlatL2 + manifest.json under data/faiss_index/
Question -> embed -> FAISS top-k -> rag.py grounded prompt -> llm.py provider
  -> answer + source attribution (filename, page, chunk id, excerpt)
```

## Module boundaries

| Module | Responsibility |
|---|---|
| `config.py` | Env-based dataclass; validates provider enums, temperature 0–2, chunk_size > chunk_overlap, top_k ≥ 1 |
| `loaders.py` | In-memory parsing; SHA-256; no OCR (scanned PDFs raise a clear error) |
| `chunking.py` | Chunking + deterministic chunk ids `<hash[:12]>-<index>` |
| `embeddings.py` | Lazy-import embedder factory; any `embed_documents`/`embed_query` object works |
| `vector_store.py` | FAISS persistence, SHA-256 dedup, embedding-profile mismatch guard |
| `llm.py` | Lazy-import provider factory (OpenAI, Anthropic, Ollama, Hugging Face) |
| `rag.py` | Strict grounding prompt, insufficient-information fallback, attribution |
| `app.py` | Streamlit orchestration only; caches the embedder via `st.cache_resource` |

Swapping the generation provider never touches ingestion/retrieval: `rag.py`
only depends on the `generate(prompt) -> str` protocol.

## Persistence and data boundary

- Original uploads are parsed in memory and are **not** copied to disk.
- Extracted chunk text + metadata **are** persisted in `data/faiss_index/manifest.json`
  alongside `index.faiss`, so the index can contain sensitive document text.
  Treat that directory as private data; it is git-ignored.
- The manifest records the embedding provider/model. On load, a mismatch
  raises `EmbeddingProfileMismatchError` and the UI asks for a rebuild,
  instead of silently querying vectors from a different embedding space.
- Only load indexes this app wrote in your own trusted directory. Never
  load an index supplied by an untrusted person.

## Performance notes

- The embedding model is built once per Streamlit process (`cache_resource`)
  and documents are embedded in batches.
- The index is built once at ingestion, not per query; duplicates are
  skipped by SHA-256 before embedding.
- Default embedding model `all-MiniLM-L6-v2` is small and CPU-friendly.
