import os
from urllib import response
import requests
from dotenv import load_dotenv
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_store"

load_dotenv()
# print("KEY:", os.getenv("GROQ_API_KEY"))

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("documents")


def query_documents(question: str, top_k: int = 3) -> dict:
    question_vector = embedder.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_vector,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append(
            {
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "chunk_index": results["metadatas"][0][i]["chunk_index"],
                "distance": round(results["distances"][0][i], 4),
            }
        )

    return {"question": question, "matches": chunks}


def build_prompt(question: str, chunks: list) -> str:
    # This is the most important part of a RAG pipeline
    # We tell the LLM exactly what to do and what NOT to do
    context = "\n\n".join(
        [
            f"[Source: {c['source']}, Chunk: {c['chunk_index']}]\n{c['text']}"
            for c in chunks
        ]
    )

    return f"""You are a helpful assistant. Answer the user's question 
using ONLY the context provided below. Do not use any outside knowledge.
If the answer is not in the context, say "I could not find this in the document."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def ask_document(question: str) -> dict:
    # Step 1 — retrieve relevant chunks
    result = query_documents(question)
    chunks = result["matches"]

    # Step 2 — build the prompt
    prompt = build_prompt(question, chunks)

    # # Step 3 — send to Ollama
    # response = requests.post(
    #     OLLAMA_URL,
    #     json={
    #         "model": OLLAMA_MODEL,
    #         "prompt": prompt,
    #         "stream": False,  # wait for full response
    #     },
    # )

    # response.raise_for_status()
    # answer = response.json()["response"].strip()

    # Step 3 — send to Groq
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
            "Content-Type": "application/json",
        },
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "stream": False,
        },
    )
    print("STATUS:", response.status_code)
    print("BODY:", response.json())

    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"].strip()

    # Step 4 — return everything
    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "distance": c["distance"],
            }
            for c in chunks
        ],
    }


# ── quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    question = input("Ask a question about your document: ")
    result = ask_document(question)

    print(f"\n{'─'*50}")
    print(f"Question : {result['question']}")
    print(f"{'─'*50}")
    print(f"Answer   : {result['answer']}")
    print(f"{'─'*50}")
    print("Sources  :")
    for s in result["sources"]:
        print(
            f"  → {s['source']} | chunk {s['chunk_index']} | distance {s['distance']}"
        )
    print(f"{'─'*50}")
