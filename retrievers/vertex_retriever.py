import logging
from clients.vertex_client import VertexClient
from retrievers.base import Retriever
from models.retrieved_context import RetrievedContext

class VertexRetriever(Retriever):
    """Application-level wrapper that adapts Vertex RAG results."""
    
    def __init__(self, client: VertexClient):
        self.client = client
        self.logger = logging.getLogger(__name__)
            
    def run_context_retrieval(self, query: str) -> list[RetrievedContext]:
        """Perform similarity search on Vertex AI search app, returing results."""        
        try:
            response = self.client.query_rag(query)
            raw_contexts = response.contexts.contexts            
            return [RetrievedContext.from_proto(proto) for proto in raw_contexts]
        except Exception as e:
            self.logger.exception(f"Error during Vertex AI Search retrieval: {e}")
            return []
    



