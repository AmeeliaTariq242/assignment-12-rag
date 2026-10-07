import os
import re
import tempfile
from collections import Counter

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(
    page_title="DocuMind RAG",
    page_icon="🧠",
    layout="wide",
)

# -------------------- Page styling --------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(145deg, #07101f 0%, #0d1830 55%, #111b2b 100%);
    }
    .banner {
        padding: 30px 34px;
        border-radius: 22px;
        background: linear-gradient(125deg, #172554, #164e63);
        border: 1px solid rgba(255,255,255,.10);
        margin-bottom: 24px;
    }
    .banner h1 { color: #ffffff; margin: 0 0 8px 0; }
    .banner p { color: #dbeafe; margin: 0; font-size: 1.03rem; }
    .hint {
        padding: 12px 16px;
        border-radius: 12px;
        background: rgba(255,255,255,.045);
        border: 1px solid rgba(255,255,255,.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="banner">
        <h1>🧠 DocuMind — Document Q&A</h1>
        <p>Build a private searchable index from your PDFs and answer questions using retrieved evidence.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- Session state --------------------
DEFAULTS = {
    "vectorstore": None,
    "pages": [],
    "chunks": [],
    "file_count": 0,
    "chunk_count": 0,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def embedding_model():
    """Load the local sentence embedding model once per app session."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )


# -------------------- Sidebar controls --------------------
with st.sidebar:
    st.header("🛠️ Workspace")

    api_key_input = st.text_input(
        "Groq API key",
        type="password",
        help="Use your own Groq key. Never commit it to GitHub.",
    )
    api_key = api_key_input or os.getenv("GROQ_API_KEY")

    st.divider()
    st.subheader("Index settings")
    chunk_size = st.slider("Chunk length", 400, 1600, 900, 100)
    overlap = st.slider("Chunk overlap", 0, 300, 120, 20)

    st.subheader("Search settings")
    top_k = st.slider("Evidence pieces", 2, 8, 5)
    candidate_count = st.slider("Search candidates", 8, 30, 16, 2)

    model_name = st.selectbox(
        "Answer model",
        ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
        index=0,
    )

    st.divider()
    if st.button("♻️ Reset workspace", use_container_width=True):
        for key in DEFAULTS:
            st.session_state[key] = DEFAULTS[key]
        st.rerun()

if not api_key:
    st.info("Enter your Groq API key in the left sidebar to activate the assistant.")
    st.stop()

os.environ["GROQ_API_KEY"] = api_key

# -------------------- Upload and indexing --------------------
st.subheader("1. Add PDF documents")
files = st.file_uploader(
    "Select one or more PDF files",
    type="pdf",
    accept_multiple_files=True,
)

if st.button("📥 Build Document Index", type="primary", use_container_width=True):
    if not files:
        st.error("Please choose at least one PDF.")
        st.stop()

    pages = []
    with st.spinner("Reading PDFs and creating searchable chunks..."):
        for uploaded in files:
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.getvalue())
                    temporary_path = tmp.name

                loaded_pages = PyPDFLoader(temporary_path).load()
                for page in loaded_pages:
                    page.metadata["source_file"] = uploaded.name
                pages.extend(loaded_pages)
            finally:
                if temporary_path and os.path.exists(temporary_path):
                    os.remove(temporary_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        chunks = splitter.split_documents(pages)

        store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model(),
        )

        st.session_state.vectorstore = store
        st.session_state.pages = pages
        st.session_state.chunks = chunks
        st.session_state.file_count = len(files)
        st.session_state.chunk_count = len(chunks)

    st.success("Document index is ready. You can now ask questions.")

# -------------------- Index status --------------------
if st.session_state.vectorstore is None:
    st.markdown(
        """
        <div class="hint">
        <b>RAG flow:</b> PDF → text extraction → chunks → embeddings → Chroma → retrieval → grounded answer
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("PDFs indexed", st.session_state.file_count)
m2.metric("Pages read", len(st.session_state.pages))
m3.metric("Chunks stored", st.session_state.chunk_count)

st.divider()

# -------------------- Hybrid retrieval helpers --------------------
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "who", "where",
    "when", "why", "how", "does", "do", "did", "in", "on", "of", "to",
    "and", "or", "for", "from", "with", "this", "that", "these", "those",
    "it", "they", "he", "she", "their", "his", "her", "tell", "me",
}


def terms(text):
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return [word for word in words if word not in STOP_WORDS and len(word) > 2]


def lexical_bonus(question, content):
    """Reward exact words/phrases from the question inside retrieved text."""
    q_terms = terms(question)
    if not q_terms:
        return 0.0

    content_lower = content.lower()
    counts = Counter(re.findall(r"[a-zA-Z0-9']+", content_lower))
    matched = sum(1 for word in q_terms if counts[word] > 0)
    score = matched / len(set(q_terms))

    # Exact multi-word phrases are especially useful for document titles.
    q_clean = re.sub(r"[^a-z0-9 ]", " ", question.lower())
    q_clean = re.sub(r"\s+", " ", q_clean).strip()
    phrase_bonus = 0.15 if len(q_clean.split()) >= 2 and q_clean in content_lower else 0.0

    return score + phrase_bonus


def retrieve_evidence(store, question, limit, final_k):
    """Combine semantic retrieval with a small lexical reranking step."""
    candidate_k = min(limit, max(final_k * 3, final_k))
    semantic = store.similarity_search_with_relevance_scores(question, k=candidate_k)

    ranked = []
    for doc, semantic_score in semantic:
        lexical = lexical_bonus(question, doc.page_content)
        combined = (0.78 * float(semantic_score)) + (0.22 * lexical)
        ranked.append((combined, doc))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in ranked[:final_k]]


# -------------------- Question answering --------------------
st.subheader("2. Ask your document")
question = st.text_input(
    "Your question",
    placeholder="Example: Who are the main characters in Crazy Paella?",
)

if st.button("🔎 Find Evidence & Answer", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question first.")
        st.stop()

    with st.spinner("Searching the document..."):
        retrieved = retrieve_evidence(
            st.session_state.vectorstore,
            question.strip(),
            candidate_count,
            top_k,
        )

    evidence_blocks = []
    for idx, doc in enumerate(retrieved, start=1):
        source = doc.metadata.get("source_file", "Unknown file")
        page = int(doc.metadata.get("page", 0)) + 1
        evidence_blocks.append(
            f"[Evidence {idx} | {source} | page {page}]\n{doc.page_content}"
        )

    context = "\n\n".join(evidence_blocks)

    prompt = f"""
You are a document-grounded question answering assistant.

Rules:
1. Use only the evidence supplied below.
2. Do not use outside knowledge to fill missing details.
3. Answer the exact question asked.
4. If the evidence does not support an answer, say exactly:
   "I could not find that answer in the uploaded document."
5. If the question names a story, section, person, or topic, make sure your answer is supported by evidence about that same subject.
6. Keep the answer concise and cite useful evidence labels such as [Evidence 1].

EVIDENCE:
{context}

QUESTION:
{question.strip()}
"""

    llm = ChatGroq(model=model_name, temperature=0)

    with st.spinner("Writing an evidence-based answer..."):
        result = llm.invoke(prompt).content

    st.subheader("💬 Answer")
    st.write(result)

    st.subheader("📚 Evidence used")
    for idx, doc in enumerate(retrieved, start=1):
        source = doc.metadata.get("source_file", "Unknown file")
        page = int(doc.metadata.get("page", 0)) + 1
        with st.expander(f"Evidence {idx} — {source} — Page {page}"):
            st.write(doc.page_content)
