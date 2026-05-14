import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class MindConfig:
    """Centralized configuration for the Continuous Learning Mind."""

    # LLM Configuration
    llm_provider: str = "openai"  # openai, anthropic, ollama, openai_compatible
    llm_model: str = "gpt-4"
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_temperature: float = 0.7
    llm_max_tokens: int = 150

    # Embedding Configuration
    embedding_provider: str = "local"  # local, openai
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_api_key: Optional[str] = None

    # Memory / Persistence
    db_path: str = "./mind_memory.db"
    similarity_threshold: float = 0.75
    max_depth: int = 3
    residue_threshold: float = 0.1

    # Personality
    personality_decay: float = 0.001
    personality_learning_rate: float = 0.05

    # Web Server
    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        return cls(
            llm_provider=os.getenv("MIND_LLM_PROVIDER", "openai"),
            llm_model=os.getenv("MIND_LLM_MODEL", "gpt-4"),
            llm_api_key=os.getenv("MIND_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            llm_base_url=os.getenv("MIND_LLM_BASE_URL"),
            llm_temperature=float(os.getenv("MIND_LLM_TEMPERATURE", "0.7")),
            llm_max_tokens=int(os.getenv("MIND_LLM_MAX_TOKENS", "150")),
            embedding_provider=os.getenv("MIND_EMBEDDING_PROVIDER", "local"),
            embedding_model=os.getenv("MIND_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            embedding_api_key=os.getenv("MIND_EMBEDDING_API_KEY"),
            db_path=os.getenv("MIND_DB_PATH", "./mind_memory.db"),
            similarity_threshold=float(os.getenv("MIND_SIMILARITY_THRESHOLD", "0.75")),
            max_depth=int(os.getenv("MIND_MAX_DEPTH", "3")),
        )
