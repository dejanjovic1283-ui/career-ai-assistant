# 🧠 AI Career Chatbot Pro

Production-ready AI career assistant built with **Streamlit, OpenAI (LLM + embeddings), and Cloud Run**.

Analyze resumes, match them with job descriptions, identify skill gaps, and get actionable career advice — all powered by LLMs.

---

## 🚀 Live Demo

👉 https://your-run-app-url.run.app

---

## ✨ Features

- 📄 Upload PDF resume
- 🧠 AI-powered career assistant (LLM)
- 🔍 Semantic search using OpenAI embeddings
- 📊 Resume ↔ Job Description matching
- 🧩 Skill gap detection
- 💬 Interactive chat interface
- ⚡ Fast deployment on Google Cloud Run

---

## 🧠 How It Works

1. Resume is uploaded (PDF)
2. Text is extracted and split into chunks
3. Chunks are converted into embeddings using OpenAI
4. User question → embedded → similarity search
5. Relevant chunks + job description → sent to LLM
6. AI returns structured career advice

---

## 🏗 Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python 3.13
- **LLM:** OpenAI (`gpt-4.1-mini`)
- **Embeddings:** OpenAI (`text-embedding-3-small`)
- **Vector Search:** NumPy (cosine similarity)
- **Deployment:** Google Cloud Run
- **Containerization:** Docker

---

## 📦 Project Structure

career-ai-assistant/ │ ├── app.py ├── requirements.txt ├── Dockerfile ├── .streamlit/ │ └── config.toml ├── .gitignore ├── .dockerignore ├── README.md

---

## ⚙️ Installation (Local)

```bash
- git clone https://github.com/dejan1283-ui/career-ai-assistant.git
- cd career-ai-assistant

- pip install -r requirements.txt

- Set environment variable:
  export OPENAI_API_KEY="your_api_key"

- Run app:
- streamlit run app.py

## ☁️ Deployment (Google Cloud Run)

1. Build & Push
- gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/career-ai
2. Deploy
--gcloud run deploy career-ai-assistant \
--image gcr.io/YOUR_PROJECT_ID/career-ai \
--platform managed \
--region europe-west1 \
--allow-unauthenticated
3. Set environment variable
OPENAI_API_KEY=your_api_key

## 💬 Example Questions

- How well does my resume match this job?
- What skills am I missing?
- How can I improve my CV?
- Suggest projects to improve my chances

## 📊 Sample Output

- Match score (0–100)
- Strengths
- Missing skills
- Resume improvements
- Actionable next steps

## 🔐 Security Notes

- API keys are stored as environment variables
- .streamlit/secrets.toml is ignored via .gitignore
- Never commit API keys to GitHub

## 🧠 Future Improvements

- 📈 Advanced scoring (semantic + structured)
- 📄 Export PDF report
- 🧩 Multi-job comparison
- 🗂 Resume versioning
- 🧠 RAG with vector DB (FAISS / Pinecone)

## 👨‍💻 Author
- Dejan — AI & Software Engineering student

## ⭐ If you like this project
- Give it a star ⭐ on GitHub and connect with me on LinkedIn!