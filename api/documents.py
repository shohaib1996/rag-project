from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from core.database import SessionLocal
from core.deps import get_current_user
from models.user import User
from models.document import Document
from core.vectorstore import delete_vectors

router = APIRouter(prefix="/api/documents", tags=["documents"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("")
def list_documents(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """
    List all documents (both files and text) uploaded by the user.
    """
    documents = (
        db.query(Document)
        .filter(Document.user_id == current_user.id)
        .order_by(Document.created_at.desc())
        .all()
    )
    return documents


@router.delete("/{document_id}")
def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Delete a document AND its associated vectors from Pinecone.
    """
    # 1. Check if document exists and belongs to user
    document = (
        db.query(Document)
        .filter(Document.id == document_id, Document.user_id == current_user.id)
        .first()
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # 2. Delete vectors from Pinecone using the document_id filter
    try:
        delete_vectors(filter={"document_id": document_id})
    except Exception as e:
        # Log this, but maybe proceed to delete from DB or raise?
        # For now, let's assume if vector delete fails, we shouldn't delete the DB record to avoid 'phantom' vectors.
        raise HTTPException(
            status_code=500, detail=f"Failed to delete vectors: {str(e)}"
        )

    # 3. Delete from Database
    db.delete(document)
    db.commit()

    return {"message": "Document deleted successfully", "id": document_id}
