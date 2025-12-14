import os
# import chromadb
# from chromadb.config import Settings

from pinecone import Pinecone, ServerlessSpec
import time
from dotenv import load_dotenv

load_dotenv()

# # Get ABSOLUTE directory safely
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DB_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "chroma_db"))

# # Make sure directory exists
# os.makedirs(DB_DIR, exist_ok=True)

# client = chromadb.PersistentClient(
#     path=DB_DIR, settings=Settings(anonymized_telemetry=False)
# )

# collection = client.get_or_create_collection(name="documents")

api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    print("❌ ERROR: PINECONE_API_KEY is not set in environment variables.")
else:
    print(f"✅ PINECONE_API_KEY found (starts with {api_key[:5]}...)")

pc = Pinecone(api_key=api_key)
index_name = "rag-app"

# Create index if not exists (Basic check)
existing_indexes = [i.name for i in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI text-embedding-3-small dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait for index to be ready
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)


def add_documents(texts, embeddings, metadatas, ids):
    # collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)

    vectors = []
    for i, text in enumerate(texts):
        # Pinecone metadata must be a dict
        meta = metadatas[i] if metadatas[i] else {}
        meta["text"] = text  # Store text in metadata for retrieval

        vectors.append({"id": ids[i], "values": embeddings[i], "metadata": meta})

    # Batch upsert is recommended (batches of 100)
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)


def query_documents(query_embedding, n_results=3, where=None):
    # return collection.query(
    #     query_embeddings=[query_embedding], n_results=n_results, where=where
    # )

    # Ensure filter is not None for Pinecone if empty
    filter_dict = where if where else {}

    return index.query(
        vector=query_embedding,
        top_k=n_results,
        include_metadata=True,
        filter=filter_dict,
    )
