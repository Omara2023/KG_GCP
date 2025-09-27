import vertexai
from vertexai.rag import retrieval_query, RagResource, RagRetrievalConfig, Filter
import logging

class VertexClient:
    """Low-level adapter for Vertex AI RAG interactions."""

    def __init__(self, project_id: str, location: str, rag_corpus_id: str):
        self.project_id = project_id
        self.location = location
        self.rag_corpus_id = rag_corpus_id
        self.logger = logging.getLogger(__name__)

        try:
            vertexai.init(project=project_id, location=location)
        except Exception as e:
            self.logger.exception(e)
            exit(1)

    def query_rag(self, text: str, top_k: int = 3, distance: float = 0.5):
        """Call Vertex RAG API and return raw response."""
        rag_corpus = f"projects/{self.project_id}/locations/{self.location}/ragCorpora/{self.rag_corpus_id}"
        rag_retrieval_config = RagRetrievalConfig(top_k=top_k, filter=Filter(vector_distance_threshold=distance))
        
        return retrieval_query(
            text=text,
            rag_resources=[RagResource(rag_corpus=rag_corpus)],
            rag_retrieval_config=rag_retrieval_config
        )

