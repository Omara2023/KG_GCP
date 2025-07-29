import os
from clients.vertex_ai_rag_client import VertexAIRagClient
from models.context import RetrievedContext

class VertexRagService:
    def __init__(self):
        self.project_id = self._get_env("GCP_PROJECT_ID")
        self.location = self._get_env("GCP_RAG_CORPUS_REGION")
        self.rag_corpus_id = self._get_env("GCP_RAG_CORPUS_ID")
        self.caller = VertexAIRagClient(self.project_id, self.location, self.rag_corpus_id)       

    def _get_env(self, var_name: str) -> str:
        value = os.environ.get(var_name)
        if not value:
            raise ValueError(f"Missing required environment variable: {var_name}")
        return value

    def search(self, query: str) -> list[RetrievedContext]:
        return self.caller.run_context_retrieval(query)