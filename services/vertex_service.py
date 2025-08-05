from clients.vertex_ai_rag_client import VertexAIRagClient
from models.context import RetrievedContext

class VertexRagService:
    def __init__(self, client: VertexAIRagClient):
        self.client = client      

    def search(self, query: str) -> list[RetrievedContext]:
        return self.client.run_context_retrieval(query)
    
