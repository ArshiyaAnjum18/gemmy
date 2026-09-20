"""Shared backend configuration loaded from environment variables."""

import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _ollama_api_url() -> str:
    configured_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api")
    normalized_url = configured_url.rstrip("/")
    if not normalized_url.endswith("/api"):
        normalized_url = f"{normalized_url}/api"
    return normalized_url


OLLAMA_BASE_URL = _ollama_api_url()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma")
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", str(PROJECT_ROOT / "backend" / "data" / "chroma")))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "backend" / "data" / "knowledge_base"
CHROMA_COLLECTION_NAME = "gemmy_knowledge_base"

HIDDEN_GEMS_EMAIL = "hiddengemssocietyngo@gmail.com"
HIDDEN_GEMS_PHONE = "+91 94946 08408"

PROGRAM_CATEGORY_ALIASES = {
    "SDP": ("student development program", "sdp"),
    "Job Readiness": ("job readiness", "job-readiness"),
    "Hackathon/Ideathon": ("hackathon", "ideathon"),
    "Corporate Training/FDP": ("corporate training", "faculty development program", "fdp"),
    "Child Development": ("child development",),
}


def detect_program_category(text: str) -> str:
    """Classify a query or explicit heading by the known program categories."""
    heading_pattern = re.compile(
        r"^\s*(?:■\s*)?program\s+\d+\s*[-–—:]\s*(student development program|job readiness program|"
        r"hackathon\s*/?\s*ideathon|corporate training|faculty development program|"
        r"child development program)",
        re.IGNORECASE,
    )
    heading_match = heading_pattern.search(text[:300])
    if heading_match:
        heading_text = heading_match.group(1).casefold()
        for category, aliases in PROGRAM_CATEGORY_ALIASES.items():
            if any(alias in heading_text for alias in aliases):
                return category

    normalized_text = text.casefold()
    for category, aliases in PROGRAM_CATEGORY_ALIASES.items():
        if any(alias in normalized_text for alias in aliases):
            return category
    return "General"
