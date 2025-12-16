from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.embeddings import embed_texts
from core.vectorstore import query_documents
from core.llm import generate_answer, generate_title
from fastapi import Depends
from core.deps import get_current_user
from models.user import User
from core.rate_limiter import rate_limiter
from core.database import SessionLocal
from models.conversation import Conversation, Message
from typing import Optional


router = APIRouter(prefix="/api/ask", tags=["ask"])


class AskRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None


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

    # documents = results.get("documents", [[]])[0]
    # metadatas = results.get("metadatas", [[]])[0]

    # Pinecone
    matches = results["matches"]
    documents = [
        match["metadata"]["text"] for match in matches if "text" in match["metadata"]
    ]
    metadatas = [match["metadata"] for match in matches]

    sources = list({meta.get("source") for meta in metadatas if meta.get("source")})

    # return {"question": payload.question, "matches": documents}

    # 3️⃣ Fallback retrieval (wider net)
    if not documents:
        results = query_documents(
            query_embedding=query_embedding,
            n_results=12,
            where={"user_id": current_user.id},
        )
        # documents = results.get("documents", [[]])[0]
        matches = results["matches"]
        documents = [
            match["metadata"]["text"]
            for match in matches
            if "text" in match["metadata"]
        ]

    # 4️⃣ Give up only if still nothing
    if not documents:
        answer = "I don't know based on the provided information."
        sources = []
    else:
        context = "\n\n".join(documents)
        answer = generate_answer(context, payload.question)

    # 4️⃣ Save Conversation & Messages
    db = SessionLocal()
    try:
        # Get or Create Conversation
        conversation_id = payload.conversation_id
        if conversation_id:
            conversation = (
                db.query(Conversation)
                .filter(
                    Conversation.id == conversation_id,
                    Conversation.user_id == current_user.id,
                )
                .first()
            )
            # If not found (e.g. invalid ID), create new
            if not conversation:
                # Generate Smart Title
                title = generate_title(payload.question)
                conversation = Conversation(user_id=current_user.id, title=title)
                db.add(conversation)
                db.commit()
                db.refresh(conversation)
                conversation_id = conversation.id
        else:
            # Generate Smart Title
            title = generate_title(payload.question)
            conversation = Conversation(user_id=current_user.id, title=title)
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
            conversation_id = conversation.id

        # Save User Message
        user_msg = Message(
            conversation_id=conversation_id, role="user", content=payload.question
        )
        db.add(user_msg)

        # Save Assistant Message
        ai_msg = Message(
            conversation_id=conversation_id, role="assistant", content=answer
        )
        db.add(ai_msg)

        db.commit()

    except Exception as e:
        print(f"Error saving conversation: {e}")
        # Don't fail the request just because saving failed
    finally:
        db.close()

    return {
        "question": payload.question,
        "answer": answer,
        "sources": sources,
        "conversation_id": conversation_id,
    }
