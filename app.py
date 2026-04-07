import os
import re
from typing import List, Optional

import numpy as np
import pdfplumber
import streamlit as st
from openai import OpenAI


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="AI Career Chatbot Pro",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .hero-card {
        background: linear-gradient(135deg, #0f172a, #1e293b, #0b1020);
        padding: 22px;
        border-radius: 18px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.20);
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.9;
    }

    .section-card {
        background: #ffffff;
        padding: 18px;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        border: 1px solid rgba(15, 23, 42, 0.06);
        height: 100%;
    }

    .small-muted {
        color: #64748b;
        font-size: 0.92rem;
    }

    .score-box {
        padding: 14px 16px;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-top: 12px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🧠 AI Career Chatbot Pro</div>
        <div class="hero-subtitle">
            Upload your resume, optionally add a job description, and chat with an AI career assistant.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================
# OpenAI setup
# =========================
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"


def load_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


# =========================
# Helpers
# =========================
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_text(uploaded_file) -> str:
    pages_text = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)
    return clean_text("\n".join(pages_text))


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_texts(texts: List[str]) -> np.ndarray:
    client = load_openai_client()

    clean_texts = [clean_text(t) for t in texts if clean_text(t)]
    if not clean_texts:
        return np.array([], dtype=np.float32)

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=clean_texts,
    )

    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float32)


@st.cache_data(show_spinner="Creating semantic index...")
def build_vector_store(chunks: List[str]):
    if not chunks:
        return None

    vectors = embed_texts(chunks)
    if vectors.size == 0:
        return None

    vectors = normalize_vectors(vectors)

    return {
        "chunks": chunks,
        "vectors": vectors,
    }


def retrieve_relevant_chunks(query: str, store, top_k: int = 5) -> List[str]:
    if not query or not store:
        return []

    query_vec = embed_texts([query])
    if query_vec.size == 0:
        return []

    query_vec = normalize_vectors(query_vec)[0]
    scores = store["vectors"] @ query_vec
    best_idx = np.argsort(scores)[::-1][:top_k]

    return [store["chunks"][i] for i in best_idx]


def build_context(
    resume_text: str,
    job_description: str,
    retrieved_chunks: List[str],
) -> str:
    context_parts = []

    if resume_text:
        context_parts.append("RESUME:\n" + resume_text[:12000])

    if job_description and clean_text(job_description):
        context_parts.append("JOB DESCRIPTION:\n" + clean_text(job_description)[:6000])

    if retrieved_chunks:
        context_parts.append("RELEVANT CHUNKS:\n" + "\n\n".join(retrieved_chunks))

    return "\n\n".join(context_parts)


def estimate_match_score(resume_text: str, job_description: str) -> Optional[int]:
    if not resume_text or not clean_text(job_description):
        return None

    resume_words = set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-\+#\.]{1,}\b", resume_text.lower()))
    jd_words = set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-\+#\.]{1,}\b", job_description.lower()))

    jd_words = {w for w in jd_words if len(w) > 2}
    if not jd_words:
        return None

    overlap = len(resume_words & jd_words)
    score = int((overlap / len(jd_words)) * 100)

    return max(15, min(score, 95))


def ask_llm(context: str, question: str, chat_history: List[dict]) -> str:
    client = load_openai_client()

    system_prompt = """
You are an expert career coach and resume assistant.

Your job is to:
- analyze the user's resume
- compare it with the job description
- identify strengths, weaknesses, and missing skills
- give specific, practical, honest advice
- keep answers clear, structured, and actionable

Rules:
- Be concise but useful.
- Use bullet points when helpful.
- If the user asks about fit, include:
  1. overall fit
  2. strengths
  3. missing skills
  4. resume improvements
  5. next steps
- Never invent resume facts that are not present in the provided context.
""".strip()

    messages = [{"role": "system", "content": system_prompt}]

    for item in chat_history[-6:]:
        messages.append({"role": item["role"], "content": item["content"]})

    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        }
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.4,
    )

    return response.choices[0].message.content or ""


# =========================
# Session state
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "job_description" not in st.session_state:
    st.session_state.job_description = ""


# =========================
# UI
# =========================
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Upload Resume")
    st.caption("Upload your resume (PDF).")

    uploaded_file = st.file_uploader(
        "Upload your resume (PDF)",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        with st.spinner("Reading your resume..."):
            resume_text = extract_pdf_text(uploaded_file)
            st.session_state.resume_text = resume_text

            chunks = chunk_text(resume_text)
            st.session_state.vector_store = build_vector_store(chunks)

        st.success("Resume uploaded and indexed successfully.")

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Optional Job Description")
    job_description = st.text_area(
        "Optional job description",
        value=st.session_state.job_description,
        placeholder="Paste the job posting here...",
        height=220,
        label_visibility="collapsed",
    )
    st.session_state.job_description = job_description
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("## Career Chat")

if st.session_state.resume_text:
    score = estimate_match_score(st.session_state.resume_text, st.session_state.job_description)
    if score is not None:
        st.markdown(
            f"""
            <div class="score-box">
                <b>Quick Match Score:</b> {score}/100
                <div class="small-muted">
                    This is a simple keyword-overlap estimate. The AI response below gives the deeper analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Ask something about your resume, job fit, or improvements...")

if user_question:
    if not st.session_state.resume_text:
        st.warning("Please upload your resume first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_question})

        retrieval_query = user_question
        if clean_text(st.session_state.job_description):
            retrieval_query += "\n\n" + clean_text(st.session_state.job_description)

        retrieved_chunks = retrieve_relevant_chunks(
            query=retrieval_query,
            store=st.session_state.vector_store,
            top_k=5,
        )

        context = build_context(
            resume_text=st.session_state.resume_text,
            job_description=st.session_state.job_description,
            retrieved_chunks=retrieved_chunks,
        )

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                answer = ask_llm(
                    context=context,
                    question=user_question,
                    chat_history=st.session_state.messages,
                )
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})