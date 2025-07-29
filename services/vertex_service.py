from clients.vertex_ai_rag_client import VertexAIRagClient
from services.mixins import EnvMixin
from models.context import RetrievedContext

class VertexRagService(EnvMixin):
    def __init__(self):
        self.project_id = self._get_env("GCP_PROJECT_ID")
        self.location = self._get_env("GCP_RAG_CORPUS_REGION")
        self.rag_corpus_id = self._get_env("GCP_RAG_CORPUS_ID")
        self.caller = VertexAIRagClient(self.project_id, self.location, self.rag_corpus_id)       

    def search(self, query: str) -> list[RetrievedContext]:
        return self.caller.run_context_retrieval(query)