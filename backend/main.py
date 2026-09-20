from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re

from backend.config import OLLAMA_BASE_URL
from backend.rag.generator import (
    OllamaConnectionError,
    OllamaResponseError,
    generate_response,
)
from backend.rag.retriever import ChromaError, EmbeddingError, retrieve

app = FastAPI(title="Gemmy Backend")
app.state.ollama_base_url = OLLAMA_BASE_URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


GREETING_PATTERN = re.compile(
    r"^(hi|hello|hey|hiya|good morning|good afternoon|good evening)[!,. ]*$",
    re.IGNORECASE,
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "Gemmy Backend",
        "message": "Use /health for health checks or /chat to send messages.",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "Gemmy Backend",
    }


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, str]:
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="message must not be empty")
    if GREETING_PATTERN.fullmatch(request.message.strip()):
        return {
            "response": (
                "Hi! I'm Gemmy, the AI assistant for Hidden Gems Society. "
                "How can I help you today?"
            )
        }

    try:
        retrieved_chunks = retrieve(request.message, top_k=3)
        context = "\n\n".join(
            f"Source: {chunk['metadata'].get('source_filename', 'unknown')}\n"
            f"Program category: {chunk['metadata'].get('category', 'General')}\n"
            f"{chunk['text'][:1200]}"
            for chunk in retrieved_chunks[:3]
        )
        if not context:
            context = "No relevant Hidden Gems knowledge was retrieved."
        response = generate_response(request.message, context=context)
    except (ChromaError, EmbeddingError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except OllamaConnectionError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except OllamaResponseError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    return {"response": response}


@app.get("/rag/search")
def rag_search(q: str = Query(..., min_length=1)) -> dict[str, object]:
    if not q.strip():
        raise HTTPException(status_code=422, detail="q must not be empty")

    try:
        chunks = retrieve(q)
    except (ChromaError, EmbeddingError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return {"query": q, "results": chunks}
