from abc import ABC, abstractmethod
from models.context import RetrievedContext
from models.llm_response import LLMResponse

class AnswerGenerator(ABC):
    """Interface for AnswerGenerator family of classes."""

    @abstractmethod
    def generate(self, query: str, contexts: list[RetrievedContext]) -> LLMResponse:
        """Query LLM using provided contexts."""
        pass

