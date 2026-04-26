import os
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_store"
DATA_PATH = "./data"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("documents")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)


def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text


def ingest_pdf(pdf_path: str) -> int:
    filename = os.path.basename(pdf_path)
    raw_text = extract_text_from_pdf(pdf_path)

    chunks = splitter.split_text(raw_text)

    embeddings = embedder.encode(chunks).tolist()

    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Ingested {len(chunks)} chunks from '{filename}'")
    return len(chunks)


if __name__ == "__main__":
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            ingest_pdf(os.path.join(DATA_PATH, file))
