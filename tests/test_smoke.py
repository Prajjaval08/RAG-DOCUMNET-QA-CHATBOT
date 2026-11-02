"""End-to-end smoke test with stub embedder + stub LLM (no paid APIs)."""

from conftest import StubEmbedder, StubLLM

from rag_chatbot.chunking import make_chunks
from rag_chatbot.loaders import load_file, sha256_hex
from rag_chatbot.rag import answer_question
from rag_chatbot.vector_store import FAISSVectorStore


def test_ingest_chunk_embed_index_retrieve_answer(tmp_path):
    data = (
        b"The capital of France is Paris. Paris is known for the Eiffel Tower.\n"
        b"Cats are independent animals that sleep for many hours each day.\n"
    )
    source_hash = sha256_hex(data)
    chunks = make_chunks(load_file("notes.txt", data), source_hash, 200, 20)
    assert chunks

    store = FAISSVectorStore(tmp_path / "index", "local", "stub-v1")
    assert store.add_chunks(chunks, StubEmbedder(), source_hash) == len(chunks)
    store.save()

    reloaded = FAISSVectorStore.load(tmp_path / "index", "local", "stub-v1")
    result = answer_question(
        "What is the capital of France?",
        reloaded,
        StubEmbedder(),
        StubLLM(answer="Paris, according to the indexed notes."),
        top_k=2,
    )
    assert "Paris" in result.answer
    assert result.sources
    assert result.sources[0].filename == "notes.txt"
    # The stub LLM saw a grounded prompt with context + question.
    llm = StubLLM()
    answer_question("cats?", reloaded, StubEmbedder(), llm, top_k=2)
    assert "CONTEXT" in llm.last_prompt
    assert "cats?" in llm.last_prompt
