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
                    page.metadata["excluded_from_retrieval"] = False
                pages.extend(loaded_pages)
            finally:
                if temporary_path and os.path.exists(temporary_path):
                    os.remove(temporary_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        # Keep answer keys / appendices out of the searchable evidence when a PDF
        # contains a clearly labelled Answer Key. This prevents a question from
        # matching the answer-key page instead of the story itself.
        answer_key_started = False
        searchable_pages = []
        for page in pages:
            page_text = page.page_content.lower()
            if "answer key" in page_text and len(page_text) < 5000:
                answer_key_started = True
            if not answer_key_started:
                searchable_pages.append(page)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        chunks = splitter.split_documents(searchable_pages)

        # Give every chunk a stable order. This lets retrieval keep nearby chunks
        # together when a question names a story/section heading.
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index

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
m3.metric("Searchable chunks", st.session_state.chunk_count)

st.divider()

# -------------------- Hybrid retrieval helpers --------------------
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "who", "where",
    "when", "why", "how", "does", "do", "did", "in", "on", "of", "to",
    "and", "or", "for", "from", "with", "this", "that", "these", "those",
    "it", "they", "he", "she", "their", "his", "her", "tell", "me",
    "main", "character", "characters", "story", "book", "document",
}


def terms(text):
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return [word for word in words if word not in STOP_WORDS and len(word) > 2]


def exact_phrases(question):
    """Return useful multi-word phrases from a question, longest first."""
    words = re.findall(r"[a-zA-Z0-9']+", question.lower())
    phrases = []
    for size in (4, 3, 2):
        for i in range(len(words) - size + 1):
            phrase = " ".join(words[i:i + size])
            if any(w not in STOP_WORDS for w in words[i:i + size]):
                phrases.append(phrase)
    return sorted(set(phrases), key=len, reverse=True)


def lexical_score(question, content):
    q_terms = terms(question)
    text = content.lower()
    tokens = set(re.findall(r"[a-zA-Z0-9']+", text))
    if not q_terms:
        return 0.0
    word_score = sum(1 for w in set(q_terms) if w in tokens) / len(set(q_terms))
    phrase_score = 0.0
    for phrase in exact_phrases(question):
        if phrase in text:
            # A named title such as "crazy paella" is much stronger evidence
            # than isolated words.
            phrase_score = max(phrase_score, min(1.0, 0.35 + 0.12 * len(phrase.split())))
    return min(1.0, 0.55 * word_score + 0.45 * phrase_score)


def retrieve_evidence(store, question, limit, final_k, all_chunks):
    """Retrieve semantic candidates, add lexical/title matches, then deduplicate pages."""
    candidate_k = min(limit, max(final_k * 5, 12))
    semantic = store.similarity_search_with_relevance_scores(question, k=candidate_k)

    # Also scan the already-indexed chunks for exact names/titles. This is cheap,
    # and it fixes cases where a distinctive story title is semantically under-ranked.
    lexical_candidates = []
    for doc in all_chunks:
        score = lexical_score(question, doc.page_content)
        if score >= 0.35:
            lexical_candidates.append((score, doc))
    lexical_candidates.sort(key=lambda x: x[0], reverse=True)

    combined = []
    seen_ids = set()
    for doc, sem_score in semantic:
        key = (doc.metadata.get("source_file"), doc.metadata.get("page"), doc.page_content[:80])
        if key not in seen_ids:
            combined.append((0.60 * float(sem_score) + 0.40 * lexical_score(question, doc.page_content), doc))
            seen_ids.add(key)
    for lex_score, doc in lexical_candidates[:max(12, final_k * 3)]:
        key = (doc.metadata.get("source_file"), doc.metadata.get("page"), doc.page_content[:80])
        if key not in seen_ids:
            combined.append((0.40 * lex_score, doc))
            seen_ids.add(key)

    # If the question contains a distinctive phrase (e.g. a story title), prefer
    # the earliest occurrence and its nearby chunks. Story headings normally mark
    # the beginning of the relevant section; this prevents answer-key matches.
    phrase_hits = []
    for phrase in exact_phrases(question):
        if len(phrase.split()) >= 2:
            for doc in all_chunks:
                if phrase in doc.page_content.lower():
                    phrase_hits.append((doc.metadata.get("chunk_index", 10**9), phrase, doc))
            if phrase_hits:
                break
    if phrase_hits:
        first_index = min(x[0] for x in phrase_hits)
        window_start = max(0, first_index - 1)
        window_end = first_index + 4
        for doc in all_chunks:
            idx = doc.metadata.get("chunk_index", -1)
            if window_start <= idx <= window_end:
                combined.append((0.95 - 0.01 * abs(idx - first_index), doc))

    combined.sort(key=lambda x: x[0], reverse=True)

    # Keep at most one chunk per page until we have enough pages. This stops the
    # UI from showing the same page five times.
    selected = []
    seen_pages = set()
    for _, doc in combined:
        page_key = (doc.metadata.get("source_file"), doc.metadata.get("page"))
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)
        selected.append(doc)
        if len(selected) >= final_k:
            break
    return selected


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
            st.session_state.chunks,
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
