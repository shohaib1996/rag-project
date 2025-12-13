from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid
from services.text_splitter import split_text
from core.embeddings import embed_texts
from core.vectorstore import add_documents
from fastapi import Depends
from core.deps import get_current_user
from models.user import User
from core.rate_limiter import rate_limiter


router = APIRouter(prefix="/api/train", tags=["train"])


class TrainTextRequest(BaseModel):
    text: str


@router.post("/text")
async def train_text(
    payload: TrainTextRequest,
    current_user: User = Depends(get_current_user),
):
    if not rate_limiter.check(current_user.id, "train"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    chunks = split_text(payload.text)
    embeddings = embed_texts(chunks)

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {
            "user_id": current_user.id,
            "source": "text",
        }
        for _ in chunks
    ]

    add_documents(
        texts=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    return {
        "message": "Text stored successfully",
        "chunks_stored": len(chunks),
    }
