# DocMind — RAG-powered Document Q&A Engine

Ask natural language questions about any PDF. 
Built with FastAPI, ChromaDB, LangChain, and Ollama.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector_store-purple)
![Ollama](https://img.shields.io/badge/Ollama-local_LLM-orange)

## Architecture


## Tech Stack

| Component       | Technology                     |
|-----------------|-------------------------------|
| Backend API     | FastAPI                        |
| Vector Database | ChromaDB                       |
| Embeddings      | sentence-transformers          |
| LLM             | Ollama (Llama 3.2, local)      |
| PDF Parsing     | PyMuPDF                        |
| Chunking        | LangChain RecursiveTextSplitter|

## Features

- Upload any PDF via drag-and-drop UI
- Semantic search across document chunks
- LLM answers grounded strictly in document context
- Source tracing — every answer shows which chunk it came from
- Fully local — no data sent to external APIs
- Clean chat UI served from FastAPI

## Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### Install

```bash
git clone https://github.com/YOUR_USERNAME/docmind.git
cd docmind
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Pull the LLM

```bash
ollama pull llama3.2
```

### Run

```bash
uvicorn main:app --reload
```

Open `http://localhost:8000` in your browser.

### API Endpoints

POST /ingest   Upload a PDF → returns chunk count
POST /query    Ask a question → returns answer + sources
GET  /docs     Swagger UI

### Example

```bash
# ingest
curl -X POST http://localhost:8000/ingest \
  -F "file=@./contract.pdf"

# query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the payment terms?"}'
```

## Project Structure

docmind/




├── ingest.py       # PDF extraction, chunking, embedding, vector storage




├── query.py        # Similarity search + LLM answer generation


├── main.py         # FastAPI server + endpoints


├── static/


│   └── index.html  # Chat UI


└── requirements.txt

## How It Works

1. **Ingestion** — PDF is parsed, split into 500-token overlapping chunks,
   embedded using sentence-transformers, stored in ChromaDB
2. **Query** — question is embedded, ChromaDB finds top-3 similar chunks
   via cosine similarity, chunks are passed to Llama 3.2 as context
3. **Answer** — LLM responds strictly from the retrieved context,
   with source chunk references returned alongside the answergit