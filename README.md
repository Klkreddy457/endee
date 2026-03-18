# RAG-Based AI Knowledge Assistant using Endee

A complete, production-ready Retrieval-Augmented Generation (RAG) pipeline built with FastAPI and the high-performance [Endee](https://github.com/endee-io/endee) vector database.

## Project Overview

This project implements an AI knowledge assistant that allows users to seamlessly upload text documents, semantically search through them, and ask complex questions based on the ingested knowledge base. The assistant leverages Endee for insanely fast vector search and OpenAI/Ollama for generating embeddings and LLM responses.

## Architecture

1. **Document Ingestion (`/upload`)**:
   - The user uploads a `.txt` document.
   - The document is parsed and split into overlapping chunks to preserve contextual meaning.
   - Dense vector embeddings are generated for each chunk using an embedding model (e.g., `text-embedding-3-small` or Ollama).
   - Vectors and metadata are stored in the **Endee** vector database.

2. **Semantic Querying & RAG (`/query`)**:
   - The user asks a question via the API.
   - The query is converted into a dense vector embedding.
   - The system queries the Endee vector database for the top-k most semantically similar document chunks.
   - The retrieved context + the user's question are packaged into a prompt and sent to an LLM (e.g., `gpt-4o-mini`).
   - The LLM streams back an accurate answer grounded strictly in the retrieved context.

## How Endee is Used

**Endee** is the core retrieval engine powering this RAG pipeline:
- **Fast Similarity Search**: Endee holds the document chunk embeddings and executes sub-millisecond similarity searches (Cosine distance) to fetch relevant context.
- **REST API Integration**: The backend communicates directly with Endee's HTTP API (`/api/v1/index/create`, `/api/v1/index/{name}/vector/insert`, and `/api/v1/index/{name}/search`).
- **Metadata Filtering**: We serialize the original text chunk and chunk metadata as JSON strings into Endee's `meta` field so documents can be instantaneously retrieved.

## Setup Instructions

### 1. Start the Endee Vector Database
You need an instance of Endee running. The easiest way is via Docker.
Run this from the root of the repository:
```bash
docker run \
  --ulimit nofile=100000:100000 \
  -p 8080:8080 \
  -v ./endee-data:/data \
  --name endee-server \
  --restart unless-stopped \
  endeeio/endee-server:latest
```
*(The server will be reachable at `http://localhost:8080`)*

### 2. Configure the Backend

Create a `.env` file in the root `endee` directory (there is a template provided):

```env
ENDEE_URL=http://localhost:8080
ENDEE_AUTH_TOKEN=
OPENAI_API_KEY=your-openai-api-key-here
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
```

*(If you wish to use a local LLM like Ollama, point `OPENAI_API_BASE` in the code to `http://localhost:11434/v1` and use your local model names).*

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the API Server
Start the FastAPI server (from the root directory):
```bash
uvicorn backend.main:app --reload --port 8000
```

## Example API Usage

The backend API is accessible at `http://localhost:8000`. It provides Swagger documentation at `http://localhost:8000/docs`.

### 1. Health Check
Check if the API and Endee are online.
```bash
curl -X GET http://localhost:8000/
```
```json
{
  "status": "healthy",
  "endee_connected": true
}
```

### 2. Upload Document (Ingestion)
Upload any `.txt` file to be chunked and indexed.
```bash
curl -X POST -F "file=@sample.txt" http://localhost:8000/upload
```
```json
{
  "message": "Successfully ingested 14 chunks.",
  "processing_time_ms": 1250.33
}
```

### 3. Ask a Question (RAG)
Query the knowledge base using the `GET` or `POST` endpoints.
```bash
curl -X GET "http://localhost:8000/query?q=What+is+Endee?"
```
```json
{
  "answer": "Endee is a high-performance open-source vector database built for AI search and retrieval workloads. It is designed for teams building RAG pipelines...",
  "sources": [
    "sample.txt"
  ],
  "processing_time_ms": 845.12
}
```
