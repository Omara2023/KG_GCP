from abc import ABC, abstractmethod

class QueryRewriter(ABC):
    """Interface for QueryRewriter family of classes."""

    @abstractmethod
    def rewrite(self, query: str) -> list[str]:
        """Rewrite user query returning a list of 1+ rewritten queries."""
        pass