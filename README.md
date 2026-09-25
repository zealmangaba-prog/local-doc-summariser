# 🔒 Local Confidential Document Summarizer

### 🐳 Run with Docker (Zero Installation)

If you have Docker and Ollama installed, run the application with a single command without configuring Python:

```bash
docker run -d -p 8501:8501 \
  --name local-doc-summarizer \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_HOST=[http://host.docker.internal:11434](http://host.docker.internal:11434) \
  <YOUR_DOCKER_USERNAME>/local-doc-summariser:latest

An air-gapped, zero-leakage document processing engine and web interface powered by Streamlit and local LLMs via Ollama. Built specifically for strictly confidential workflows where documents must never touch the cloud or write unencrypted buffers to disk.

---

## Architecture & Security Highlights

* **In-Memory Streaming:** Document parsing runs purely within ephemeral RAM buffers using Python's `io.BytesIO`. No intermediate text files or temp files touch the storage drive.
* **Semantic Map-Reduce:** Long-form documents are chunked using `RecursiveCharacterTextSplitter` and synthesized via local Ollama models (`llama3.2`).
* **Offline Operational Boundary:** Zero external API calls, tracking, or network egress.
* **Test Isolation:** Full unit test suite with mocked inference runs offline in under 0.3s.

---

## Tech Stack

* **Frontend:** Streamlit
* **Extraction & Chunking:** `pypdf`, `langchain-text-splitters`
* **Inference Runtime:** Ollama (`llama3.2`)
* **Testing & Quality Assurance:** `pytest`, `pytest-mock`, `pylint`

---

## Getting Started

### 1. Prerequisites
Ensure [Ollama](https://ollama.com/) is installed and running:
```bash
ollama pull llama3.2
