from fastapi import FastAPI
from dotenv import load_dotenv
import os

from core.database import Base, engine
from models import user  # noqa: F401 Ensure models are loaded
from api.train import router as train_router
from api.ask import router as ask_router
from api.train_file import router as train_file_router
from api.auth import router as auth_router

load_dotenv()

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="RAG Backend")

app.include_router(train_router)
app.include_router(ask_router)
app.include_router(train_file_router)
app.include_router(auth_router)


@app.get("/health")
def health():
    return {"status": "ok", "env_loaded": bool(os.getenv("OPENAI_API_KEY"))}
