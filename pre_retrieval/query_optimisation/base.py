from abc import ABC, abstractmethod
from models.user_query import UserQuery

class QueryRewriter(ABC):
    """Interface for QueryRewriter family of classes."""

    @abstractmethod
    def rewrite_query(self, query: UserQuery) -> list[UserQuery]:
        """Rewrite user query returning a list of 1+ rewritten queries."""
        pass