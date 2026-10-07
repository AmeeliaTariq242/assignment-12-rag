import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(
    page_title="Document AI - RAG Assistant",
    page_icon="📚",
    layout="wide",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #07111f 0%, #0b1730 55%, #101b2e 100%);
    }
    .hero {
        padding: 28px 32px;
        border-radius: 20px;
        background: linear-gradient(120deg, #172554, #0f3b4d);
        border: 1px solid rgba(255,255,255,.10);
        margin-bottom: 22px;
    }
    .hero h1 {
        color: white;
        margin: 0 0 8px 0;
    }
    .hero p {
        color: #cbd5e1;
        margin: 0;
        font-size: 1.05rem;
    }
    .source-card {
        padding: 14px 16px;
        margin: 10px 0;
        border-radius: 12px;
        background: rgba(255,255,255,.05);
        border: 1px solid rgba(255,255,255,.10);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>📚 Document AI — RAG Assistant</h1>
        <p>Upload a PDF, build a searchable knowledge base, retrieve relevant evidence,
        and ask questions grounded in your document.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Session state ----------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "documents" not in st.session_state:
    st.session_state.documents = []
if "chunks" not in st.session_state:
    st.session_state.chunks = 0
if "files" not in st.session_state:
    st.session_state.files = 0

# ---------- Cached embedding model ----------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ RAG Settings")

    api_key_input = st.text_input(
        "Groq API Key",
        type="password",
        help="Enter your Groq API key. You can also store GROQ_API_KEY in a .env file.",
    )
    api_key = api_key_input or os.getenv("GROQ_API_KEY")

    st.divider()

    chunk_size = st.slider("Chunk size", 400, 1600, 900, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 120, 20)
    top_k = st.slider("Retrieved chunks", 2, 8, 4)

    model_name = st.selectbox(
        "Groq model",
        ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
    )

    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        st.session_state.vectorstore = None
        st.session_state.documents = []
        st.session_state.chunks = 0
        st.session_state.files = 0
        st.rerun()

if not api_key:
    st.info("Enter your Groq API key in the sidebar to start.")
    st.stop()

os.environ["GROQ_API_KEY"] = api_key

# ---------- Upload ----------
st.subheader("1. Upload your document")

uploaded_files = st.file_uploader(
    "Choose one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

# ---------- Build knowledge base ----------
if st.button("🔨 Build Knowledge Base", type="primary", use_container_width=True):
    if not uploaded_files:
        st.error("Please upload at least one PDF file.")
        st.stop()

    all_pages = []

    with st.spinner("Loading and processing your documents..."):
        for uploaded_file in uploaded_files:
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                    temp.write(uploaded_file.getvalue())
                    temp_path = temp.name

                pages = PyPDFLoader(temp_path).load()

                for page in pages:
                    page.metadata["source_file"] = uploaded_file.name

                all_pages.extend(pages)
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_documents(all_pages)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
        )

        st.session_state.vectorstore = vectorstore
        st.session_state.documents = all_pages
        st.session_state.chunks = len(chunks)
        st.session_state.files = len(uploaded_files)

    st.success("Knowledge base created successfully.")

# ---------- Status ----------
if st.session_state.vectorstore is not None:
    c1, c2, c3 = st.columns(3)
    c1.metric("PDF Files", st.session_state.files)
    c2.metric("Pages", len(st.session_state.documents))
    c3.metric("Text Chunks", st.session_state.chunks)

    st.divider()

    # ---------- Ask ----------
    st.subheader("2. Ask a question")

    question = st.text_input(
        "Question",
        placeholder="Example: What is the main idea of this document?",
    )

    if st.button("🔎 Retrieve Evidence & Generate Answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k,
                "fetch_k": max(12, top_k * 4),
            },
        )

        retrieved_docs = retriever.invoke(question)

        evidence = []
        context_parts = []

        for i, doc in enumerate(retrieved_docs, start=1):
            source = doc.metadata.get("source_file", "Unknown file")
            page = doc.metadata.get("page", 0) + 1

            evidence.append((i, source, page, doc.page_content))
            context_parts.append(
                f"[Source {i} | {source} | Page {page}]\n{doc.page_content}"
            )

        context = "\n\n".join(context_parts)

        llm = ChatGroq(
            model=model_name,
            temperature=0,
        )

        prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the evidence supplied below.
Do not invent facts or use outside knowledge.
If the evidence does not contain the answer, say:
"I could not find that answer in the uploaded document."

Use simple, clear language.
When useful, cite the evidence using labels such as [Source 1].

DOCUMENT EVIDENCE:
{context}

QUESTION:
{question}
"""

        with st.spinner("Generating a grounded answer..."):
            response = llm.invoke(prompt)

        st.subheader("🤖 Answer")
        st.write(response.content)

        st.subheader("📌 Retrieved Evidence")

        for i, source, page, text in evidence:
            with st.expander(f"Source {i} — {source} — Page {page}"):
                st.write(text)

else:
    st.markdown(
        """
        ### How this application works

        **PDF → Load → Chunk → Embed → Store → Retrieve → Generate**

        1. Upload a PDF.
        2. The PDF is loaded page by page.
        3. Text is divided into smaller chunks.
        4. Each chunk is converted into an embedding.
        5. Embeddings are stored in Chroma.
        6. Your question retrieves the most relevant chunks.
        7. Groq generates an answer from the retrieved evidence.
        """
    )
