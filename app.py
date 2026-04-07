import os
import re
from typing import List

import faiss
import numpy as np
import pdfplumber
import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="AI Career Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------
# Styling
# ---------------------------
st.markdown(
    """
    <style>
        .hero-card {
            background: linear-gradient(135deg, #0f172a, #1e293b);
            padding: 28px;
            border-radius: 22px;
            color: white;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
            margin-bottom: 20px;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.4rem;
        }
        .hero-subtitle {
            font-size: 1rem;
            color: #cbd5e1;
        }
        .soft-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            margin-bottom: 16px;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.7rem;
        }
        .tag {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            margin: 0.15rem 0.25rem 0.15rem 0;
            border-radius: 999px;
            background: #eef2ff;
            color: #312e81;
            font-size: 0.9rem;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Session state
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

if "job_description" not in st.session_state:
    st.session_state.job_description = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

# ---------------------------
# Cached models
# ---------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)

# ---------------------------
# Helpers
# ---------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_chunks(text: str, chunk_size: int = 700, overlap: int = 120) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += max(1, chunk_size - overlap)
    return chunks


def build_vector_store(chunks):
    if not chunks:
        return None

    model = load_embedding_model()

    embeddings = model.encode(chunks, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index


def retrieve_relevant_chunks(query: str, chunks, index, k: int = 4):
    if not chunks or index is None:
        return []

    model = load_embedding_model()
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding.astype("float32")

    distances, indices = index.search(query_embedding, k)

    results = []
    for i in indices[0]:
        if i < len(chunks):
            results.append(chunks[i])

    return results


def build_context(resume_text: str, job_description: str, retrieved_chunks: List[str]) -> str:
    context_parts = []

    if resume_text:
        context_parts.append("RESUME:\n" + resume_text[:4000])

    if job_description.strip():
        context_parts.append("JOB DESCRIPTION:\n" + job_description[:2500])

    if retrieved_chunks:
        context_parts.append("RELEVANT CHUNKS:\n" + "\n\n".join(retrieved_chunks))

    return "\n\n".join(context_parts)


def ask_llm(context: str, question: str, chat_history: List[dict]) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system_prompt = """
You are an expert career coach and resume assistant.

You help users understand their resume, improve it, compare it with job descriptions,
and answer career-related questions.

Rules:
- Be practical, specific, and encouraging.
- Use the provided resume and job description context.
- If information is missing, say so clearly.
- Give structured answers when helpful.
- Do not invent resume details that are not present.
"""

    messages = [{"role": "system", "content": system_prompt}]

    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}"
        }
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.4,
    )

    return response.choices[0].message.content


# ---------------------------
# Layout
# ---------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🤖 AI Career Chatbot Pro</div>
        <div class="hero-subtitle">
            Upload your resume, optionally add a job description, and chat with an AI career assistant.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload Resume</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Optional Job Description</div>', unsafe_allow_html=True)
    jd_text = st.text_area(
        "Paste a target job description",
        placeholder="Paste the job posting here...",
        height=170,
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    with st.spinner("Reading your resume and preparing the chatbot context..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        chunks = split_into_chunks(resume_text)

        combined_chunks = chunks.copy()
        if jd_text.strip():
            combined_chunks.extend(split_into_chunks(jd_text))

        index = build_vector_store(combined_chunks)

        st.session_state.resume_text = resume_text
        st.session_state.job_description = jd_text
        st.session_state.chunks = combined_chunks
        st.session_state.index = index

    st.success("Resume loaded. You can start chatting below.")

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Suggested Questions</div>', unsafe_allow_html=True)
    suggestions = [
        "What are the strongest parts of my resume?",
        "What skills seem to be missing for this job?",
        "How can I improve my summary section?",
        "Am I a strong match for this role?",
        "Write 3 bullet improvements for my Projects section.",
    ]
    st.markdown(
        "".join([f'<span class="tag">{q}</span>' for q in suggestions]),
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.subheader("Career Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask something about your resume, job fit, or improvements...")

if prompt:
    if not st.session_state.resume_text:
        st.warning("Please upload a PDF resume first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retrieved = retrieve_relevant_chunks(
                    prompt,
                    st.session_state.chunks,
                    st.session_state.index,
                    k=4,
                )

                context = build_context(
                    st.session_state.resume_text,
                    st.session_state.job_description,
                    retrieved,
                )

                answer = ask_llm(context, prompt, st.session_state.messages[:-1])
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

if st.session_state.resume_text:
    with st.expander("Show extracted resume text"):
        st.text(st.session_state.resume_text[:5000])