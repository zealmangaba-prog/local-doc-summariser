import streamlit as st
from summariser_engine import DocumentSummarizer, DocumentProcessingError

st.set_page_config(
    page_title="Confidential Document Summarizer",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔒 Internal Document Summarizer")
st.caption(
    "Air-gapped, zero-leakage local document analysis. No data leaves this device."
)

with st.sidebar:
    st.header("Model Settings")
    selected_model = st.selectbox(
        "Local LLM Model",
        options=["llama3.2", "llama3.1", "mistral", "qwen2.5"],
        index=0,
        help="Ensure the selected model has been pulled locally in Ollama.",
    )

    chunk_size = st.slider(
        "Chunk Character Size",
        min_value=800,
        max_value=3000,
        value=1500,
        step=100,
        help="Target length of individual text segments sent to the local model.",
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Number of overlapping characters between adjacent chunks to maintain context.",
    )

    st.divider()
    st.markdown(
        """
        **Security Guarantee:**
        - Files buffered strictly in RAM via `io.BytesIO`.
        - No network egress or third-party APIs.
        """
    )

uploaded_file = st.file_uploader(
    "Upload Confidential PDF Document",
    type=["pdf"],
    help="Select a local PDF containing readable digital text.",
)

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Filename:** `{uploaded_file.name}`")
    with col2:
        st.info(f"**File Size:** `{uploaded_file.size / 1024:.2f} KB`")

    if st.button("Summarize Document", type="primary", use_container_width=True):
        try:
            summarizer = DocumentSummarizer(
                model_name=selected_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            with st.spinner("Processing document locally via Ollama..."):
                file_bytes = uploaded_file.read()
                summary = summarizer.summarize(file_bytes)

            st.success("Summary Generated Successfully!")
            st.subheader("Generated Summary")
            st.markdown(summary)

            st.download_button(
                label="Download Summary (.txt)",
                data=summary,
                file_name=f"summary_{uploaded_file.name.replace('.pdf', '')}.txt",
                mime="text/plain",
            )

        except DocumentProcessingError as e:
            st.error(f"Document Processing Error: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")