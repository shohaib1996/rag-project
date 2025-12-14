import uuid
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from core.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    filename = Column(String, nullable=False)  # For text, this can be a snippet title
    file_type = Column(String, nullable=False)  # 'pdf', 'docx', 'text'
    created_at = Column(DateTime(timezone=True), server_default=func.now())
