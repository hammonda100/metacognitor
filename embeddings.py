import numpy as np
import asyncio
from abc import ABC, abstractmethod
from typing import List, Union


class BaseEmbeddingProvider(ABC):
    """Abstract base for embedding providers."""

    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        pass


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """Local sentence-transformers provider."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI API embeddings provider."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self._client = None

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        import openai
        if self._client is None:
            self._client = openai.OpenAI(api_key=self.api_key)
        resp = self._client.embeddings.create(input=texts, model=self.model)
        return np.array([d.embedding for d in resp.data])


class EmbeddingClient:
    """Unified embedding client with async support."""

    def __init__(self, config):
        if config.embedding_provider == "local":
            self.provider = SentenceTransformerProvider(config.embedding_model)
        elif config.embedding_provider == "openai":
            key = config.embedding_api_key or config.llm_api_key
            self.provider = OpenAIEmbeddingProvider(key, config.embedding_model)
        else:
            raise ValueError(f"Unknown embedding provider: {config.embedding_provider}")

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        result = self.provider.embed(texts)
        if isinstance(texts, str):
            if result.ndim == 2:
                return result[0]
            return result
        return result

    async def aembed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Async wrapper that runs embedding in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed, texts)
