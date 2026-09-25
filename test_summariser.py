import pytest
from unittest.mock import patch, MagicMock
from summariser_engine import DocumentSummarizer, DocumentProcessingError


@pytest.fixture
def summarizer():
    """Provides a default DocumentSummarizer instance for testing."""
    return DocumentSummarizer(model_name="llama3.2", chunk_size=500, chunk_overlap=50)


def test_empty_byte_stream_raises_error(summarizer):
    """Ensure an empty byte buffer immediately triggers DocumentProcessingError."""
    with pytest.raises(DocumentProcessingError, match="Uploaded file buffer is empty."):
        summarizer.extract_text(b"")


def test_split_text_into_chunks_structure(summarizer):
    """Verify semantic chunking splits long strings without losing continuity."""
    sample_text = "Standard corporate disclosure statement. " * 30
    chunks = summarizer.split_text_into_chunks(sample_text)

    assert isinstance(chunks, list)
    assert len(chunks) > 1
    # Verify chunks adhere to specified boundary bounds
    for chunk in chunks:
        assert len(chunk) <= 600


@patch("ollama.chat")
def test_summarize_chunks_mocked(mock_ollama_chat, summarizer):
    """Verify Ollama inference parsing without hitting the local GPU daemon."""
    mock_ollama_chat.return_value = {
        "message": {"content": "Executive bullet point summary."}
    }

    test_chunks = ["Clause 1: Confidentiality.", "Clause 2: Severability."]
    results = summarizer.summarize_chunks(test_chunks)

    assert len(results) == 2
    assert results[0] == "Executive bullet point summary."
    assert mock_ollama_chat.call_count == 2


@patch("ollama.chat")
def test_single_chunk_fast_path(mock_ollama_chat, summarizer):
    """Documents fitting in a single chunk should skip the Reduce step."""
    mock_ollama_chat.return_value = {
        "message": {"content": "Concise single chunk summary."}
    }

    # Mock extract_text and split_text_into_chunks to return a single item
    with patch.object(summarizer, "extract_text", return_value="Short agreement."), \
         patch.object(summarizer, "split_text_into_chunks", return_value=["Short agreement."]):
        
        final_summary = summarizer.summarize(b"fake pdf content")

        assert final_summary == "Concise single chunk summary."
        # Call count should be exactly 1 (no reduce step needed)
        assert mock_ollama_chat.call_count == 1