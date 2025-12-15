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
from core.database import SessionLocal
from models.document import Document


router = APIRouter(prefix="/api/train", tags=["train"])


class TrainTextRequest(BaseModel):
    text: str


# ... (imports)


@router.post("/text")
async def train_text(
    payload: TrainTextRequest,
    current_user: User = Depends(get_current_user),
):
    if not rate_limiter.check(current_user.id, "train"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # 1. Create Document record
    snippet = payload.text[:50].replace("\n", " ").strip() + "..."
    doc_entry = Document(
        user_id=current_user.id, filename=f"Text: {snippet}", file_type="text"
    )

    db = SessionLocal()
    try:
        db.add(doc_entry)
        db.commit()
        db.refresh(doc_entry)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

    # 2. Process Vectors
    chunks = split_text(payload.text)
    embeddings = embed_texts(chunks)

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {
            "user_id": current_user.id,
            "source": "text",  # Kept for backward compat if needed
            "document_id": doc_entry.id,  # New Link
            "text": chunk,  # Already added in vectorstore.py but explicitly good here too
        }
        for chunk in chunks  # FIX: iterate over chunks, not range if we want the actual chunk text available easily here
    ]

    add_documents(
        texts=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    return {
        "message": "Text stored successfully",
        "document_id": doc_entry.id,
        "chunks_stored": len(chunks),
    }
