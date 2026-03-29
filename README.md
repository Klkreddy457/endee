# 🧠 Endee RAG — AI Knowledge Assistant

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-required-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge)

**A production-ready Retrieval-Augmented Generation (RAG) pipeline** built on top of the [Endee](https://github.com/EndeeLabs/endee) open-source vector database.

Upload documents → Ask questions → Get AI-powered answers grounded in your knowledge base.

[Demo](#web-ui) · [Quick Start](#quick-start) · [API Docs](#api-reference)

</div>

---

## ✨ Features

- 📄 **Multi-format ingestion** — `.txt`, `.pdf`, `.docx`, `.md`, `.csv`, `.json`
- ⚡ **High-speed vector search** — powered by Endee's cosine similarity engine
- 🧩 **Local embeddings** — `sentence-transformers/all-MiniLM-L6-v2` (no API key, no cost)
- 🤖 **LLM answers** — via [OpenRouter](https://openrouter.ai) free tier (`gpt-4o-mini`)
- 💬 **Conversation memory** — session-based multi-turn chat
- 🔄 **Async ingestion** — background job processing with real-time status polling
- 🖥️ **Built-in Web UI** — clean chat interface served at `/`

---

## 🏗️ Architecture

```
┌─────────────┐     Upload      ┌──────────────────────┐     Vectors     ┌─────────────┐
│    User     │ ─────────────► │  FastAPI  (port 8000) │ ──────────────► │  Endee DB   │
│  (Browser)  │                 │                       │                 │ (port 8080) │
│             │ ◄───────────── │  /query               │ ◄────────────── │             │
└─────────────┘    AI Answer    └──────────┬────────────┘  Top-K chunks  └─────────────┘
                                           │
                               ┌───────────▼────────────┐
                               │   OpenRouter (LLM)     │
                               │   gpt-4o-mini          │
                               └────────────────────────┘
```

**Ingestion flow:** File → Parse → Chunk → Embed (local) → Store in Endee

**Query flow:** Question → Embed (local) → Search Endee → LLM → Answer

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker Desktop (for Endee vector DB)
- Free [OpenRouter](https://openrouter.ai) API key (for LLM)

---

### Step 1 — Clone & build the Endee DB image

```bash
git clone https://github.com/YOUR_USERNAME/endee.git
cd endee

# Build the Endee vector DB image from source (~5-10 min, first time only)
docker build -f infra/Dockerfile -t endee-oss:latest .

# Start Endee DB in the background
docker-compose up -d
```

> ✅ Endee DB is now running at `http://localhost:8080`

---

### Step 2 — Configure environment

Create a `.env` file in the project root:

```env
ENDEE_URL=http://localhost:8080
ENDEE_AUTH_TOKEN=

# Get a free key at https://openrouter.ai
OPENAI_API_KEY=sk-or-v1-your-key-here
OPENAI_API_BASE=https://openrouter.ai/api/v1
LLM_MODEL=openai/gpt-4o-mini

# Embeddings are generated locally — no key needed
EMBEDDING_MODEL=openai/text-embedding-3-small
```

---

### Step 3 — Install Python dependencies

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
pip install sentence-transformers
```

---

### Step 4 — Run the FastAPI server

```bash
uvicorn backend.main:app --port 8000 --reload
```

Open **http://localhost:8000** 🎉

---

## 🐳 Running with Docker

This project uses Docker to run the **Endee vector database**. The FastAPI backend runs locally alongside it. Choose one of the two options below.

---

### Option A — Use the pre-built Docker Hub image *(recommended, fastest)*

No compilation needed — pulls the official image directly:

```bash
docker run \
  --ulimit nofile=100000:100000 \
  -p 8080:8080 \
  -v ./endee-data:/data \
  --name endee-server \
  --restart unless-stopped \
  -d \
  endeeio/endee-server:latest
```

> ✅ Endee DB will be live at `http://localhost:8080` within seconds.

---

### Option B — Build from source *(for development / custom builds)*

Compiles the Endee C++ server inside Docker (~5–10 min, first time only):

```bash
# From the project root — build the image
docker build -f infra/Dockerfile -t endee-oss:latest .

# Start using docker-compose
docker-compose up -d
```

`docker-compose.yml` starts a container named `endee-oss` that:
- Exposes the vector DB at **http://localhost:8080**
- Persists data in the named Docker volume `endee-data`
- Restarts automatically unless manually stopped

---

### Verify Endee is healthy

```bash
docker ps                                        # confirm container is running
curl http://localhost:8080/api/v1/health         # should return {"status":"ok"}
```

### Start the FastAPI backend (separate terminal)

```bash
# Activate your virtual environment first
# Windows:
.\venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

uvicorn backend.main:app --port 8000 --reload
```

Open **http://localhost:8000** 🎉

### Stop the Endee container

```bash
# Option A (docker run):
docker stop endee-server && docker rm endee-server

# Option B (docker-compose):
docker-compose down        # data is preserved in the Docker volume
docker-compose down -v     # ⚠️ WARNING: also deletes all indexed vector data
```

---

## 🖥️ Web UI

The app ships with a built-in chat interface:

- **Upload panel** — drag & drop documents to index
- **Chat panel** — ask questions and get cited answers
- **Documents tab** — manage your knowledge base
- **Engine tab** — view system info

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Health check |
| `POST` | `/upload` | Upload & index a document |
| `GET` | `/jobs/{job_id}` | Poll ingestion job status |
| `GET` | `/documents` | List indexed documents |
| `DELETE` | `/documents/{filename}` | Remove document from registry |
| `POST` | `/query` | Ask a question (JSON body) |
| `GET` | `/query?q=...` | Ask a question (URL param) |
| `DELETE` | `/session/{id}` | Clear conversation history |

> 📖 Interactive Swagger docs: **http://localhost:8000/docs**

---

### Example — Upload a document

```bash
curl -X POST -F "file=@my-document.pdf" http://localhost:8000/upload
```
```json
{
  "message": "'my-document.pdf' accepted. Ingestion running in background.",
  "job_id": "3f9c1a2b-...",
  "status": "queued"
}
```

### Example — Ask a question

```bash
curl "http://localhost:8000/query?q=What+is+this+document+about?"
```
```json
{
  "answer": "The document discusses...",
  "sources": ["my-document.pdf"],
  "processing_time_ms": 412.3,
  "session_id": "abc-123"
}
```

---

## 📁 Project Structure

```
endee/
├── backend/
│   ├── main.py            # FastAPI app & all routes
│   ├── rag_pipeline.py    # Ingestion + query orchestration
│   ├── embedding.py       # Local embeddings + OpenRouter LLM
│   ├── endee_client.py    # HTTP client for Endee vector DB
│   ├── document_loader.py # File parsers & recursive text chunker
│   ├── reranker.py        # Cross-encoder reranker (sentence-transformers)
│   ├── config.py          # Settings loaded from .env
│   ├── utils.py           # Shared logger
│   └── static/            # Web UI (index.html)
├── infra/
│   └── Dockerfile         # Multi-stage build: compiles Endee from C++ source
├── src/                   # Endee C++ source code
├── docker-compose.yml     # Runs Endee DB on port 8080
├── requirements.txt       # Python dependencies
└── .env                   # Your secrets (not committed)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| API Server | FastAPI + Uvicorn |
| Vector Database | [Endee](https://github.com/EndeeLabs/endee) (self-hosted, open-source) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (local, free) |
| LLM | OpenRouter → `gpt-4o-mini` |
| Document Parsing | PyMuPDF · python-docx · built-in CSV/JSON |
| Containerization | Docker + docker-compose |

---

## 🔁 Restarting After a Reboot

```bash
# Start Endee DB
docker-compose up -d

# Start FastAPI (in a new terminal)
.\venv\Scripts\activate
uvicorn backend.main:app --port 8000 --reload
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

[Apache 2.0](LICENSE) — built on the [Endee](https://github.com/EndeeLabs/endee) open-source vector database.
