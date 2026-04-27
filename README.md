---
title: DocMind
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# DocMind — RAG-powered Document Q&A Engine

Ask natural language questions about any PDF using semantic search and a local LLM — no data sent to external APIs.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector_store-purple)
![Groq](https://img.shields.io/badge/Groq-Llama_3.1-orange)

🚀 **Live Demo:** [huggingface.co/spaces/ranajoy19/docmind](https://huggingface.co/spaces/ranajoy19/docmind)

---

## Architecture

    PDF → PyMuPDF → Chunker → Embedder (all-MiniLM-L6-v2) → ChromaDB
                                                                   ↓
    Question → Embedder → Similarity Search → Groq (Llama 3.1) → Answer + Sources

---

## Tech Stack

| Component       | Technology                          |
|-----------------|-------------------------------------|
| Backend API     | FastAPI                             |
| Vector Database | ChromaDB                            |
| Embeddings      | sentence-transformers (all-MiniLM-L6-v2) |
| LLM             | Groq — Llama 3.1 8B Instant         |
| PDF Parsing     | PyMuPDF                             |
| Chunking        | LangChain RecursiveCharacterTextSplitter |
| Frontend        | Vanilla HTML/CSS/JS (served by FastAPI) |
| Deployment      | Hugging Face Spaces (Docker)        |

---

## Features

- Upload any PDF through a clean chat UI
- Semantic search across document chunks — finds meaning, not just keywords
- LLM answers grounded strictly in document context — no hallucination
- Source tracing — every answer shows exactly which chunk it came from
- Cosine similarity distance scores returned with every result
- REST API with Swagger docs at `/docs`

---

## How It Works

**Phase 1 — Ingestion (once per document)**

1. PyMuPDF extracts raw text from the uploaded PDF
2. LangChain splits text into 500-token overlapping chunks (50-token overlap)
3. sentence-transformers embeds each chunk into a 384-dimension vector
4. ChromaDB stores vectors + text + metadata (source filename, chunk index) on disk

**Phase 2 — Query (every question)**

1. The question is embedded using the same sentence-transformer model
2. ChromaDB finds the top 3 most semantically similar chunks via cosine similarity
3. Chunks are passed to Groq (Llama 3.1) as grounded context
4. LLM returns a natural language answer strictly based on the retrieved chunks
5. Answer + source references are returned to the UI

---

## Quick Start (local)

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com) (free, no credit card)

### Install

```bash
git clone https://github.com/ranajoy19/docmind.git
cd docmind
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```
GROQ_API_KEY=your-groq-key-here
```

### Run

```bash
uvicorn main:app --reload
```

Open `http://localhost:8000` in your browser.

---

## API Endpoints

| Method | Endpoint  | Description                          |
|--------|-----------|--------------------------------------|
| GET    | `/`       | Chat UI                              |
| POST   | `/ingest` | Upload a PDF → returns chunk count   |
| POST   | `/query`  | Ask a question → answer + sources    |
| GET    | `/docs`   | Swagger UI                           |

### Example

```bash
# ingest a PDF
curl -X POST http://localhost:8000/ingest \
  -F "file=@./contract.pdf"

# response
{ "status": "success", "file": "contract.pdf", "chunks_stored": 47 }

# ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the payment terms?"}'

# response
{
  "question": "What are the payment terms?",
  "answer": "According to the document, payment terms are net 30 days from invoice date...",
  "sources": [
    { "source": "contract.pdf", "chunk_index": 14, "distance": 0.2341 },
    { "source": "contract.pdf", "chunk_index": 15, "distance": 0.3812 }
  ]
}
```

---

## Project Structure

    docmind/
    ├── ingest.py        # PDF extraction, chunking, embedding, vector storage
    ├── query.py         # Similarity search + Groq LLM answer generation
    ├── main.py          # FastAPI server, endpoints, static file serving
    ├── static/
    │   └── index.html   # Chat UI (vanilla HTML/CSS/JS)
    ├── Dockerfile       # Hugging Face Spaces deployment
    ├── requirements.txt
    └── .env             # GROQ_API_KEY (not committed)

---

## Deployment

Deployed on **Hugging Face Spaces** using Docker.

The app runs on port `7860` as required by HF Spaces. The Groq API key is injected via HF Repository Secrets — no keys are hardcoded or committed.

To deploy your own instance:

1. Fork this repo
2. Create a new HF Space (SDK: Docker)
3. Push this repo to your Space:
```bash
git remote add space https://YOUR_HF_USERNAME:YOUR_HF_TOKEN@huggingface.co/spaces/YOUR_HF_USERNAME/docmind
git push space main
```
4. Add `GROQ_API_KEY` in your Space → Settings → Repository Secrets

---

## License

MIT