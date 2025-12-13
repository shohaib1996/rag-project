# from langchain_text_splitters import CharacterTextSplitter
# from chromadb import PersistentClient
# from sentence_transformers import SentenceTransformer
# from openai import OpenAI
# import os
# from dotenv import load_dotenv

# load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# def load_and_split():
#     with open("./backend/data/companyPolicies.txt", "r", encoding="utf-8") as f:
#         text = f.read()

#     splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

#     chunks = splitter.split_text(text)
#     return chunks


# def create_embeddings_and_store(chunks):
#     # 1. Load embedding model
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     embeddings = model.encode(chunks).tolist()  # convert to list for Chroma

#     # 2. Initialize ChromaDB client (NEW API)
#     client = PersistentClient(path="./backend/chroma_db")

#     # 3. Create or get a collection
#     collection = client.get_or_create_collection(name="company_policies")

#     # 4. Insert chunks into Chroma
#     for idx, chunk in enumerate(chunks):
#         collection.add(ids=[str(idx)], documents=[chunk], embeddings=[embeddings[idx]])

#     print("Embeddings stored successfully!")
#     return collection


# def retrieve(query, top_k=3):
#     model = SentenceTransformer("all-MiniLM-L6-v2")

#     query_embedding = model.encode([query]).tolist()
#     client = PersistentClient(path="./backend/chroma_db")
#     collection = client.get_collection("company_policies")

#     results = collection.query(
#         query_embeddings=query_embedding,
#         n_results=top_k,
#     )

#     return results


# def answer_question(query, tok_k=3):
#     results = retrieve(query, top_k=tok_k)
#     relevent_chunk = results["documents"][0]

#     context = "\n".join(relevent_chunk)

#     prompt = f"""
#     You are an assistant. Answer the user's question ONLY using the context below.
#     If the answer is not in the context, say "I don't know."

#     Context: {context}

#     Question: {query}

#     Answer:
#     """

#     completion = client.chat.completions.create(
#         model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
#     )

#     return completion.choices[0].message.content


# if __name__ == "__main__":
#     chunks = load_and_split()
#     # print(f"Total chunks: {len(chunks)}")
#     # for i, ch in enumerate(chunks):
#     #     print(f"\n--- Chunk {i + 1} ---")
#     #     print(ch)
#     collection = create_embeddings_and_store(chunks)
#     # print("\nTesting Retrieval:")
#     query = "What is the smoking policy?"
#     # results = retrieve(query)
#     answer = answer_question(query)
#     print("Answer: ", answer)

#     # print(results["documents"])

from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv
import time

APP_START = time.time()

# -----------------------------
# Load ENV + OpenAI Client
# -----------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# GLOBAL INITIALIZATION (FAST ⚡)
# -----------------------------
print("Loading embedding model...")
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to ChromaDB...")
CHROMA_CLIENT = PersistentClient(path="./backend/chroma_db")

try:
    COLLECTION = CHROMA_CLIENT.get_collection("company_policies")
    print("Loaded existing collection.")
except Exception:
    COLLECTION = None
    print("No collection found yet.")

print("Startup Time:", time.time() - APP_START)


# -----------------------------
# Load + Split Document (Better)
# -----------------------------
def load_and_split():
    with open("./backend/data/companyPolicies.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # NEW splitter (fixes chunking issues)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10,
        separators=["\n\n", "\n", ". ", " "],
    )

    return splitter.split_text(text)


# -----------------------------
# Create embeddings ONLY ONCE
# -----------------------------
def create_embeddings_and_store(chunks):
    global COLLECTION

    if COLLECTION is None:
        COLLECTION = CHROMA_CLIENT.get_or_create_collection("company_policies")

        print("Creating embeddings...")
        embeddings = EMBED_MODEL.encode(chunks).tolist()

        print("Storing embeddings in ChromaDB...")
        for idx, chunk in enumerate(chunks):
            COLLECTION.add(
                ids=[str(idx)],
                documents=[chunk],
                embeddings=[embeddings[idx]],
            )

        print("Embeddings stored successfully!")
    else:
        print("Embeddings already exist — Skipping.")

    return COLLECTION


# -----------------------------
# Fast Retriever (No Reloads)
# -----------------------------
def retrieve(query, top_k=3):
    query_embedding = EMBED_MODEL.encode([query]).tolist()

    results = COLLECTION.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    return results


# -----------------------------
# Generate Answer Using GPT-4o-mini
# -----------------------------
def answer_question(query, top_k=3):
    # ---------------------------
    # 1. Retrieval timing
    # ---------------------------
    start = time.time()
    results = retrieve(query, top_k=top_k)
    retrieve_time = time.time() - start
    print("Retrieve Time:", retrieve_time)

    relevant_chunks = results["documents"][0]
    context = "\n\n".join(relevant_chunks)

    # Strict anti-hallucination prompt
    prompt = f"""
Using ONLY the information in the context, provide a SHORT and CLEAR answer.

Rules:
- First, identify the key facts from the context.
- Then give a concise final answer (2–3 sentences maximum).
- Do NOT add information that is not inside the context.
- Do NOT repeat or restate large parts of the context.
- If the answer is not found in the context, say "I don't know."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    # ---------------------------
    # 2. GPT timing
    # ---------------------------
    start = time.time()
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}],
    )
    gpt_time = time.time() - start
    print("GPT Time:", gpt_time)

    return completion.choices[0].message.content, retrieve_time, gpt_time


# -----------------------------
# MAIN TEST RUN (Total Execution Time)
# -----------------------------
if __name__ == "__main__":
    total_start = time.time()

    # ❗ Only run these ONCE, not every time
    # chunks = load_and_split()
    # create_embeddings_and_store(chunks)

    answer, ret_t, gpt_t = answer_question("What is the smoking policy?")

    print("\nAnswer:", answer)

    total_time = time.time() - total_start
    print(f"\nTotal Execution Time: {total_time} seconds")
