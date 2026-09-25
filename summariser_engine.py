import io 
from typing import List, Union
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# Custom exception to isolate document ingestion and parsing errors cleanly
# without terminating the entire application runtime.
class DocumentProcessingError(Exception):
    """Raised when an uploaded document cannot be read, parsed, or processed."""
    pass


# Encapsulates all summarization workflows in an object-oriented class
# to allow easy model swaps, chunk tuning, and deterministic pipeline re-use.
class DocumentSummarizer:
    def __init__(
        self,
        model_name: str = "llama3.2",
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
    ):
        """
        Initializes the local summarizer engine.

        - model_name: Targets the local quantized Ollama model.
        - chunk_size (1500 chars): Fits within the optimal attention span of 8B models,
          preventing context saturation and loss of fine legal nuance.
        - chunk_overlap (200 chars): Retains cross-boundary context between consecutive
          chunks so key terms and clauses are not split abruptly mid-sentence.
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text(self, source: Union[str, bytes]) -> str:
        """
        Extracts digital text from either a file path or raw memory bytes.

        io.BytesIO is used when handling 'bytes' to keep confidential documents
        strictly in RAM (e.g., from web uploaders), ensuring no unencrypted
        temporary files are ever written to the local disk.
        """
        try:
            if isinstance(source, bytes):
                if not source:
                    raise DocumentProcessingError("Uploaded file buffer is empty.")
                # Read directly from RAM buffer
                stream = io.BytesIO(source)
                reader = PdfReader(stream)
            else:
                # Read from direct filesystem path
                reader = PdfReader(source)

            extracted_pages: List[str] = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_pages.append(text)

            consolidated = "\n\n".join(extracted_pages).strip()

            # Guardrail: Fail fast if the document is a scanned image or text-less PDF
            if not consolidated:
                raise DocumentProcessingError(
                    "No extractable text found. The document may be a scanned image "
                    "or an encrypted PDF without an accessible text layer."
                )

            return consolidated

        except Exception as exc:
            if isinstance(exc, DocumentProcessingError):
                raise
            raise DocumentProcessingError(f"PDF extraction failure: {str(exc)}") from exc

    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Splits lengthy texts into smaller, overlapping segments.

        RecursiveCharacterTextSplitter is chosen because it attempts splits on
        natural semantic boundaries (double line breaks, single line breaks, spaces)
        before falling back to raw character breaks, preventing garbled sentences.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        return text_splitter.split_text(text)

    def summarize_chunks(self, chunks: List[str]) -> List[str]:
        """
        Executes the 'Map' stage: processes each individual text chunk locally
        using Ollama to produce targeted, high-density intermediate summaries.
        """
        summaries: List[str] = []
        for index, chunk in enumerate(chunks, start=1):
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise corporate document analyst. "
                            "Extract and summarize core policies, obligations, terms, "
                            "and constraints. Do not hallucinate or add assumptions."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Summarize section {index}:\n\n{chunk}",
                    },
                ],
            )
            # Accesses the message content directly from Ollama's response object
            summaries.append(response["message"]["content"])

        return summaries

    def summarize(self, source: Union[str, bytes]) -> str:
        """
        Executes the complete Map-Reduce summarization pipeline.

        - If document is short (1 chunk): returns the initial summary directly.
        - If document is long (multiple chunks): performs a 'Reduce' pass, where
          the local LLM synthesizes all chunk summaries into one consolidated,
          structured executive brief.
        """
        text = self.extract_text(source)
        chunks = self.split_text_into_chunks(text)
        chunk_summaries = self.summarize_chunks(chunks)

        # Single-chunk fast path: no reduce step needed
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]

        # Reduce step: Combine intermediate findings into a structured legal brief
        combined_summaries = "\n\n".join(chunk_summaries)
        final_response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an executive legal and policy advisor. Your task is "
                        "to synthesize multiple section summaries into a single cohesive "
                        "executive document brief. Use clear headings: "
                        "Executive Overview, Key Obligations & Policies, and Potential Risks."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Synthesize these section summaries:\n\n{combined_summaries}",
                },
            ],
        )

        return final_response["message"]["content"]