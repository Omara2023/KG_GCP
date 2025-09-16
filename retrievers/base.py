from abc import ABC, abstractmethod
from models.retrieved_context import RetrievedContext

class Retriever(ABC):
    """Base RAG retriever."""

    @abstractmethod
    def run_context_retrieval(self, query: str) -> list[RetrievedContext]:
        """Perform similarity search on Vertex AI search app, returing results."""
        pass

