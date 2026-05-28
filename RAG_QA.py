"""RAG_Q&A Streamlit app

Usage:
    streamlit run RAG_Q&A.py

Requirements (suggested):
    pip install streamlit langchain_community langchain-text-splitters sentence-transformers faiss-cpu PyPDF2

Notes:
- This app uses local Hugging Face embeddings (sentence-transformers/all-MiniLM-L6-v2) and FAISS.
- Optionally uses Ollama if available for LLM-based answer generation. You can also plug in an OpenAI LLM implementation if desired.
"""

import os
import re
import streamlit as st
from pathlib import Path
from typing import List

# LangChain / helpers
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Optional LLM (Ollama)
try:
    from langchain_community.llms import Ollama
    _OLLAMA_AVAILABLE = True
except Exception:
    _OLLAMA_AVAILABLE = False

# -----------------------------
# Utilities
# -----------------------------

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    return text.strip()


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


# -----------------------------
# Core pipeline functions
# -----------------------------

def load_documents(folder: str) -> List:
    loader = DirectoryLoader(
        folder,
        glob=["*.pdf", "*.txt", "*.docx"],
        loader_cls=lambda path: (
            PyPDFLoader(path)
            if path.lower().endswith('.pdf')
            else Docx2txtLoader(path)
            if path.lower().endswith('.docx')
            else TextLoader(path)
        ),
    )
    docs = loader.load()
    return docs


def preprocess_documents(docs):
    cleaned = [clean_text(d.page_content) for d in docs]
    return cleaned


def split_documents(texts, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def build_vectorstore(chunks, embeddings_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


@st.cache_resource
def load_vectorstore(index_folder: str, emb_model: str):
    """Load FAISS index with caching.

    The result is cached by Streamlit based on the input arguments so the
    index load is performed only once per process (or when arguments change).
    """
    # Note: allow_dangerous_deserialization=True may be required for some FAISS files.
    # Only enable this if you trust the index files' origin.
    embeddings = HuggingFaceEmbeddings(model_name=emb_model)
    vs = FAISS.load_local(index_folder, embeddings, allow_dangerous_deserialization=True)
    return vs


def retrieve_docs(vectorstore, query: str, k: int = 5):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    # Support both runnable (.invoke) and classic (.get_relevant_documents)
    try:
        docs = retriever.invoke(query)
    except Exception:
        try:
            docs = retriever.get_relevant_documents(query)
        except Exception:
            docs = []
    return docs


def generate_answer_with_ollama(llm, query: str, docs: List):
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}
"""
    resp = llm.invoke(prompt)
    return resp


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="RAG Q&A", layout="wide")
st.title("RAG Q&A — Retrieval-Augmented Generation (local)")

# Sidebar config
st.sidebar.header("Configuration 🔧")
base_dir = st.sidebar.text_input("Document folder", value=os.path.join(os.getcwd(), "pdf_data"))
chunk_size = st.sidebar.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=100)
chunk_overlap = st.sidebar.number_input("Chunk overlap", min_value=0, max_value=500, value=100, step=10)
emb_model = st.sidebar.text_input("Embeddings model", value="sentence-transformers/all-MiniLM-L6-v2")
index_folder = st.sidebar.text_input("FAISS index folder", value="faiss_index")
use_ollama = st.sidebar.checkbox("Use Ollama LLM (if available)", value=_OLLAMA_AVAILABLE)
k = st.sidebar.slider("Retriever k (num docs)", 1, 10, 3)

ensure_dir(index_folder)

# Main controls
col1, col2 = st.columns([2, 1])
with col1:
    st.header("Ask a question")
    query = st.text_input("Your question", value="What is sensor fusion?")
    run_query = st.button("Run Query")

with col2:
    st.header("Index & Docs")
    if st.button("Load documents"):
        with st.spinner("Loading documents..."):
            docs = load_documents(base_dir)
            st.success(f"Loaded {len(docs)} documents")
            st.session_state['docs'] = docs

    if st.button("Build index (from docs)"):
        if 'docs' not in st.session_state or not st.session_state['docs']:
            st.error("No documents loaded. Click 'Load documents' first.")
        else:
            with st.spinner("Preprocessing, splitting and building vectorstore..."):
                texts = preprocess_documents(st.session_state['docs'])
                chunks = split_documents(texts, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                st.session_state['chunks'] = chunks
                vs = build_vectorstore(chunks, embeddings_model_name=emb_model)
                vs.save_local(index_folder)
                st.session_state['vectorstore'] = vs
                st.success(f"Built and saved index with {len(chunks)} chunks")

    if st.button("Load existing index"):
        try:
            vs = load_vectorstore(index_folder, emb_model)
            st.session_state['vectorstore'] = vs
            st.success("Loaded FAISS index (cached)")
        except Exception as e:
            st.error(f"Failed to load index: {e}")

    if st.button("Clear session state"):
        for kkey in ['docs', 'chunks', 'vectorstore']:
            if kkey in st.session_state:
                del st.session_state[kkey]
        st.success("Cleared session state")

# Run query
if run_query:
    if 'vectorstore' not in st.session_state:
        st.error("No vectorstore available. Build or load an index first.")
    else:
        with st.spinner("Retrieving relevant documents..."):
            docs = retrieve_docs(st.session_state['vectorstore'], query, k=k)
        st.subheader("Retrieved chunks 📚")
        for i, d in enumerate(docs, 1):
            st.markdown(f"**Chunk {i}:**")
            st.write(d.page_content[:1000])

        # Generate a final answer
        if use_ollama and _OLLAMA_AVAILABLE:
            with st.spinner("Generating answer with Ollama..."):
                llm = Ollama(model="mistral")
                ans = generate_answer_with_ollama(llm, query, docs)
            st.subheader("Answer 🧠")
            st.write(ans)
        else:
            st.subheader("Answer (retrieval only)")
            st.write("\n\n".join([d.page_content for d in docs[:3]]))
            if use_ollama and not _OLLAMA_AVAILABLE:
                st.warning("Ollama not installed or not available. Install and run Ollama to enable LLM answers.")

# Footer / quick tip
st.markdown("---")
st.write("**Tip:** Use '**Build index**' after loading or updating documents. To use an LLM, enable Ollama in the sidebar and make sure Ollama is installed and running.")
