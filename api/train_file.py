from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
import os
import uuid
import shutil
from services.file_loader import extract_text_from_file
from services.text_splitter import split_text
from core.embeddings import embed_texts
from core.vectorstore import add_documents
from core.deps import get_current_user
from models.user import User
from core.rate_limiter import rate_limiter


router = APIRouter(prefix="/api/train_file", tags=["train_file"])

UPLOAD_DIR = "./temp_uploads"
ALLOWED_TYPES = [
    "text/plain",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
]

os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/file")
async def train_file(
    file: UploadFile = File(...), current_user: User = Depends(get_current_user)
):
    if not rate_limiter.check(current_user.id, "train"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type")

    file_id = str(uuid.uuid4())
    file_ext = file.filename.split(".")[-1]
    temp_path = os.path.join(UPLOAD_DIR, file_id + "." + file_ext)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        text = extract_text_from_file(temp_path, file_ext)

        if not text.strip():
            raise HTTPException(status_code=400, detail="File is empty")

        chunks = split_text(text)

        embeddings = embed_texts(chunks)

        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

        metadatas = [
            {"source": file.filename, "user_id": current_user.id}
            for _ in range(len(chunks))
        ]

        add_documents(texts=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)

    finally:
        os.remove(temp_path)

    return {"message": "Text extracted successfully", "characters": len(text)}
