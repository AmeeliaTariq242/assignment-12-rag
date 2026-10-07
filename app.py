
import os
import re
import tempfile
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

load_dotenv()

st.set_page_config(
    page_title="DocuMind — Document Q&A",
    page_icon="🧠",
    layout="wide",
)

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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="banner">
        <h1>🧠 DocuMind — Document Q&A</h1>
        <p>Turn your PDFs into a searchable knowledge base and answer questions from retrieved evidence.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

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
def load_sentence_model():
    """Load the local embedding model once and reuse it."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


class LocalEmbeddings(Embeddings):
    """Small LangChain adapter around SentenceTransformer."""

    def __init__(self, model):
        self.model = model

    def _encode(self, texts: List[str]):
        if not texts:
            raise ValueError("No text was supplied for embedding.")
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if getattr(vectors, "ndim", 1) == 1:
            vectors = vectors.reshape(1, -1)
        return vectors.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0]


@st.cache_resource(show_spinner=False)
def embedding_function():
    return LocalEmbeddings(load_sentence_model())


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

        if not pages:
            st.error("No readable PDF pages were found.")
            st.stop()

        # Keep clearly labelled answer-key material out of the evidence pool.
        # This is especially useful for books that place an answer key at the end.
        searchable_pages = []
        answer_key_started = False
        for page in pages:
            text = page.page_content.strip()
            lower = text.lower()
            if "answer key" in lower and len(text) < 5000:
                answer_key_started = True
            if not answer_key_started:
                searchable_pages.append(page)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        chunks = splitter.split_documents(searchable_pages)

        if not chunks:
            st.error(
                "The PDF was loaded, but no searchable text chunks were created. "
                "Try another PDF or reduce the chunk size."
            )
            st.stop()

        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index

        try:
            store = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_function(),
            )
        except Exception as exc:
            st.error(
                "The document index could not be created. "
                "The PDF was read, but the embedding model did not return valid vectors."
            )
            st.caption(f"Technical detail: {type(exc).__name__}: {exc}")
            st.stop()

        st.session_state.vectorstore = store
        st.session_state.pages = pages
        st.session_state.chunks = chunks
        st.session_state.file_count = len(files)
        st.session_state.chunk_count = len(chunks)

    st.success("Document index is ready. You can now ask questions.")

if st.session_state.vectorstore is None:
    st.info("RAG flow: PDF → Load → Chunk → Embed → Store → Retrieve → Answer")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("PDFs indexed", st.session_state.file_count)
m2.metric("Pages read", len(st.session_state.pages))
m3.metric("Searchable chunks", st.session_state.chunk_count)

st.divider()

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "who", "where",
    "when", "why", "how", "does", "do", "did", "in", "on", "of", "to",
    "and", "or", "for", "from", "with", "this", "that", "these", "those",
    "it", "they", "he", "she", "their", "his", "her", "tell", "me",
    "main", "character", "characters", "story", "book", "document",
}


def tokens(text):
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return [word for word in words if word not in STOP_WORDS and len(word) > 2]


def phrases(text):
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    result = []
    for size in (4, 3, 2):
        for i in range(len(words) - size + 1):
            part = words[i:i + size]
            if any(w not in STOP_WORDS for w in part):
                result.append(" ".join(part))
    return sorted(set(result), key=lambda x: (len(x.split()), len(x)), reverse=True)


def lexical_score(question, content):
    q = set(tokens(question))
    text = content.lower()
    if not q:
        return 0.0

    word_score = sum(1 for word in q if word in text.split()) / len(q)

    phrase_score = 0.0
    for phrase in phrases(question):
        if phrase in text:
            phrase_score = max(
                phrase_score,
                min(1.0, 0.35 + 0.12 * len(phrase.split())),
            )

    return min(1.0, 0.55 * word_score + 0.45 * phrase_score)


def best_title_anchor(question, chunks):
    """
    Find the most useful occurrence of a named multi-word phrase.
    Prefer actual section/story starts over table-of-contents mentions.
    """
    multi = [p for p in phrases(question) if len(p.split()) >= 2]
    if not multi:
        return None

    for phrase in multi:
        matches = []
        for doc in chunks:
            text = doc.page_content.lower()
            if phrase not in text:
                continue

            idx = doc.metadata.get("chunk_index", 10**9)
            page = int(doc.metadata.get("page", 0)) + 1

            score = 0.0
            # Story/section headings are stronger anchors.
            if re.search(rf"(?m)^\s*{re.escape(phrase)}\s*$", text):
                score += 5.0
            if "chapter 1" in text:
                score += 4.0
            if page >= 10:
                score += 1.0
            # Very early pages are often contents/title pages.
            if page <= 8:
                score -= 3.0

            matches.append((score, -page, idx, doc))

        if matches:
            matches.sort(reverse=True, key=lambda x: (x[0], x[1], -x[2]))
            return matches[0][2], phrase

    return None


def retrieve_evidence(store, question, limit, final_k, chunks):
    candidate_k = min(limit, max(final_k * 5, 12))
    semantic = store.similarity_search_with_relevance_scores(
        question,
        k=candidate_k,
    )

    combined = []
    seen = set()

    for doc, score in semantic:
        key = (
            doc.metadata.get("source_file"),
            doc.metadata.get("page"),
            doc.page_content[:100],
        )
        if key in seen:
            continue
        seen.add(key)
        combined.append(
            (0.60 * float(score) + 0.40 * lexical_score(question, doc.page_content), doc)
        )

    # Add exact lexical matches.
    lexical = []
    for doc in chunks:
        score = lexical_score(question, doc.page_content)
        if score >= 0.35:
            lexical.append((score, doc))

    lexical.sort(key=lambda x: x[0], reverse=True)
    for score, doc in lexical[:max(12, final_k * 3)]:
        key = (
            doc.metadata.get("source_file"),
            doc.metadata.get("page"),
            doc.page_content[:100],
        )
        if key not in seen:
            seen.add(key)
            combined.append((0.40 * score, doc))

    # If a question contains a distinctive story/section title, anchor retrieval
    # around the real section start rather than a table of contents entry.
    anchor = best_title_anchor(question, chunks)
    if anchor:
        anchor_index, _ = anchor
        for doc in chunks:
            idx = doc.metadata.get("chunk_index", -1)
            if anchor_index - 1 <= idx <= anchor_index + 5:
                combined.append(
                    (0.98 - 0.015 * abs(idx - anchor_index), doc)
                )

    combined.sort(key=lambda x: x[0], reverse=True)

    # One evidence chunk per page first, so the final context is diverse.
    selected = []
    seen_pages = set()

    for _, doc in combined:
        page_key = (
            doc.metadata.get("source_file"),
            doc.metadata.get("page"),
        )
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)
        selected.append(doc)
        if len(selected) >= final_k:
            break

    return selected


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

    if not retrieved:
        st.warning("No relevant evidence was retrieved.")
        st.stop()

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
2. Do not use outside knowledge.
3. Answer the exact question.
4. If the evidence does not support the answer, say:
   "I could not find that answer in the uploaded document."
5. If a story or section is named, use evidence about that same story or section.
6. Do not combine characters or facts from unrelated sections.
7. Keep the answer concise and cite evidence labels such as [Evidence 1].

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
