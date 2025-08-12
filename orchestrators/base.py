from abc import ABC, abstractmethod
from models.llm_response import LLMResponse

class BaseRAGOrchestrator(ABC):
    """Interface for a complete RAG-pipeline."""

    @abstractmethod
    async def generate(self, query: str) -> LLMResponse: #consider changing return type to terminalllmresponse
        pass