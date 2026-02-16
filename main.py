from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
import asyncio
import httpx
from core.database import Base, engine
from models import user, conversation  # noqa: F401 Ensure models are loaded
from api.train import router as train_router
from api.ask import router as ask_router
from api.train_file import router as train_file_router
from api.auth import router as auth_router
from api.documents import router as documents_router
from api.conversations import router as conversations_router

# Load environment variables
load_dotenv()


# Create database tables
Base.metadata.create_all(bind=engine)

RENDER_URL = os.getenv("RENDER_EXTERNAL_URL", "")
PING_INTERVAL = 10 * 60  # 10 minutes in seconds


async def keep_alive():
    """Periodically ping the server to prevent Render free tier from sleeping."""
    if not RENDER_URL:
        return
    async with httpx.AsyncClient() as client:
        while True:
            await asyncio.sleep(PING_INTERVAL)
            try:
                await client.get(f"{RENDER_URL}/health")
            except Exception:
                pass


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(keep_alive())
    yield
    task.cancel()


# Initialize FastAPI app
app = FastAPI(
    title="RAG Backend",
    description="Knowledge-based RAG backend API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://rag-project-t4mo.onrender.com",
        "https://rag-documents-mu.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(train_router)
app.include_router(ask_router)
app.include_router(train_file_router)
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(conversations_router)


# Root route (important for Railway / browsers)
@app.get("/")
def root():
    return {
        "message": "RAG Backend is running",
        "health": "/health",
        "docs": "/docs",
    }


# Health check
@app.get("/health")
def health():
    return {
        "status": "ok",
        "env_loaded": bool(os.getenv("OPENAI_API_KEY")),
    }
