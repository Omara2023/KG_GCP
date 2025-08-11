from abc import ABC, abstractmethod
from pydantic import BaseModel

class Context(BaseModel, ABC):
    """Models any RAG context to be fed to LLM in addition to a query."""
    pass

    @abstractmethod
    def to_user_friendly(self) -> str:
        """Return a human-readable representation of the context."""
        pass