import aiohttp
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class BaseLLMProvider(ABC):
    """Abstract base for all LLM providers."""

    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        pass

    @abstractmethod
    async def generate(self, prompt: str, num_residues: int = 3, **kwargs) -> List[str]:
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI and OpenAI-compatible API provider."""

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 150):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://api.openai.com"
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"OpenAI API error {resp.status}: {text}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    async def generate(self, prompt: str, num_residues: int = 3, **kwargs) -> List[str]:
        messages = [{"role": "user", "content": prompt}]
        content = await self.chat(messages, max_tokens=50 * num_residues, **kwargs)
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        return lines[:num_residues] if lines else [f"{prompt}_idea_{i}" for i in range(num_residues)]


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229",
                 temperature: float = 0.7, max_tokens: int = 150):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        url = "https://api.anthropic.com/v1/messages"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Anthropic API error {resp.status}: {text}")
                data = await resp.json()
                return data["content"][0]["text"]

    async def generate(self, prompt: str, num_residues: int = 3, **kwargs) -> List[str]:
        messages = [{"role": "user", "content": prompt}]
        content = await self.chat(messages, max_tokens=50 * num_residues, **kwargs)
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        return lines[:num_residues] if lines else [f"{prompt}_idea_{i}" for i in range(num_residues)]


class OllamaProvider(BaseLLMProvider):
    """Local Ollama provider."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434",
                 temperature: float = 0.7):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": kwargs.get("temperature", self.temperature)},
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Ollama API error {resp.status}: {text}")
                data = await resp.json()
                return data["message"]["content"]

    async def generate(self, prompt: str, num_residues: int = 3, **kwargs) -> List[str]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": kwargs.get("temperature", self.temperature)},
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Ollama API error {resp.status}: {text}")
                data = await resp.json()
                lines = [l.strip() for l in data["response"].splitlines() if l.strip()]
                return lines[:num_residues] if lines else [f"{prompt}_idea_{i}" for i in range(num_residues)]


class LLMClient:
    """Unified LLM client that routes to the configured provider."""

    def __init__(self, config):
        provider = config.llm_provider.lower()
        if provider in ("openai", "openai_compatible"):
            self.provider = OpenAIProvider(
                api_key=config.llm_api_key or "",
                model=config.llm_model,
                base_url=config.llm_base_url,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens
            )
        elif provider == "anthropic":
            self.provider = AnthropicProvider(
                api_key=config.llm_api_key or "",
                model=config.llm_model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens
            )
        elif provider == "ollama":
            self.provider = OllamaProvider(
                model=config.llm_model,
                base_url=config.llm_base_url or "http://localhost:11434",
                temperature=config.llm_temperature
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        return await self.provider.chat(messages, **kwargs)

    async def generate(self, prompt: str, **kwargs) -> List[str]:
        return await self.provider.generate(prompt, **kwargs)
