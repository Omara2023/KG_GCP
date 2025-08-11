from abc import ABC, abstractmethod
from models.context import Context
from models.llm_response import LLMResponse

class AnswerGenerator(ABC):
    """Interface for AnswerGenerator family of classes."""

    @abstractmethod
    def generate(self, query: str, contexts: list[Context]) -> LLMResponse:
        """Query LLM using provided contexts."""
        pass

    def _ensure_type(self, contexts: list[Context], expected_type: type) -> None:
        if not contexts:
            raise ValueError("No contexts provided.")
        if not all(isinstance(c, expected_type) for c in contexts):
            raise ValueError(f"Expected all contexts to be {expected_type.__name__}, "
                         f"got {type(contexts[0]).__name__}")

