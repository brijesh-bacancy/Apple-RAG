"""Runtime settings, read from environment / .env."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    ollama_base_url: str
    ollama_model: str
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    gemini_api_key: str
    gemini_model: str
    embedding_model: str
    rerank_model: str
    top_k: int
    rerank_top_k: int
    rerank_mode: str  # none | prose | all
    use_vision: bool
    index_dir: Path
    vision_cache: Path


def get_settings() -> Settings:
    """Build Settings from environment variables, with defaults tuned for this 12-page report."""
    e = os.getenv
    return Settings(
        llm_provider=e("LLM_PROVIDER", "extractive").lower(),
        ollama_base_url=e("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=e("OLLAMA_MODEL", "llama3.2:3b"),
        openai_api_key=e("OPENAI_API_KEY", ""),
        openai_base_url=e("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_model=e("OPENAI_MODEL", "gpt-4o-mini"),
        gemini_api_key=e("GEMINI_API_KEY", ""),
        gemini_model=e("GEMINI_MODEL", "gemini-flash-lite-latest"),
        embedding_model=e("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        rerank_model=e("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        top_k=int(e("TOP_K", "15")),
        rerank_top_k=int(e("RERANK_TOP_K", "5")),
        rerank_mode=e("RERANK_MODE", "all").lower(),
        use_vision=e("USE_VISION", "1") == "1",
        index_dir=BASE_DIR / "index",
        vision_cache=BASE_DIR / "data" / "vision_cache.json",
    )
