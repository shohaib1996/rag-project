from fastapi import FastAPI
from pydantic import BaseModel
import time

# Import your RAG functions
from backend.rag import answer_question

app = FastAPI()


# -----------------------------
# API Request Body Model
# -----------------------------
class Query(BaseModel):
    question: str
    top_k: int = 3


# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
def root():
    return {"message": "RAG API is running!"}


# -----------------------------
# /ask → Main RAG Endpoint
# -----------------------------
@app.post("/ask")
def ask_rag(query: Query):
    start_total = time.time()

    # Use your existing RAG answer function
    answer, retrieve_time, gpt_time = answer_question(query.question, top_k=query.top_k)

    total_time = time.time() - start_total

    return {
        "question": query.question,
        "answer": answer,
        "retrieve_time": retrieve_time,
        "gpt_time": gpt_time,
        "total_time": total_time,
    }
