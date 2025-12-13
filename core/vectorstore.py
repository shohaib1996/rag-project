import os
import chromadb
from chromadb.config import Settings

# Get ABSOLUTE directory safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "chroma_db"))

# Make sure directory exists
os.makedirs(DB_DIR, exist_ok=True)

client = chromadb.PersistentClient(
    path=DB_DIR, settings=Settings(anonymized_telemetry=False)
)

collection = client.get_or_create_collection(name="documents")


def add_documents(texts, embeddings, metadatas, ids):
    collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)


def query_documents(query_embedding, n_results=3, where=None):
    return collection.query(
        query_embeddings=[query_embedding], n_results=n_results, where=where
    )
