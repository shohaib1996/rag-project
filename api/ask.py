from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.embeddings import embed_texts
from core.vectorstore import query_documents
from core.llm import generate_answer
from fastapi import Depends
from core.deps import get_current_user
from models.user import User
from core.rate_limiter import rate_limiter


router = APIRouter(prefix="/api/ask", tags=["ask"])


class AskRequest(BaseModel):
    question: str


@router.post("")
async def ask_question(
    payload: AskRequest, current_user: User = Depends(get_current_user)
):
    if not rate_limiter.check(current_user.id, "ask"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    query_embedding = embed_texts([payload.question])[0]

    results = query_documents(
        query_embedding=query_embedding, n_results=6, where={"user_id": current_user.id}
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    sources = list({meta.get("source") for meta in metadatas if meta.get("source")})

    # return {"question": payload.question, "matches": documents}

    # 3️⃣ Fallback retrieval (wider net)
    if not documents:
        results = query_documents(
            query_embedding=query_embedding,
            n_results=12,
            where={"user_id": current_user.id},
        )
        documents = results.get("documents", [[]])[0]

    # 4️⃣ Give up only if still nothing
    if not documents:
        return {
            "question": payload.question,
            "answer": "I don't know based on the provided information.",
            "sources": [],
        }

    context = "\n\n".join(documents)

    answer = generate_answer(context, payload.question)

    return {"question": payload.question, "answer": answer, "sources": sources}
