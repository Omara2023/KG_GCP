from abc import ABC, abstractmethod
from pydantic import BaseModel

class LLMResponse(BaseModel, ABC):
    """Abstract base class to model all responses from LLMs."""
    pass

    @abstractmethod
    def to_user_friendly(self) -> str:
        """Return a human-readable representation of the response."""
        pass

