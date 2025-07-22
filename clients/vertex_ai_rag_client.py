import logging
import vertexai
from vertexai import rag
from models.context import Context

class VertexAIRagClient:
    """Wrapper class that calls Vertex AI RAG engine to retrieve context."""

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
            
    def run_context_retrieval(self, query: str) -> list[Context] | None:
        """Perform similarity search on Vertex AI search app, returing results."""        
        rag_retrieval_config = self._rag_retrieval_config()
        rag_corpus = self._rag_corpus_resoruce()
        try:
            response = rag.retrieval_query(
                text=query,
                rag_resources=[
                    rag.RagResource(
                        rag_corpus=rag_corpus 
                    )
                ],
                rag_retrieval_config=rag_retrieval_config
            )
                    
            raw_contexts = response.contexts.contexts            
            return [Context.from_proto(proto) for proto in raw_contexts]
        except Exception as e:
            self.logger.exception(f"Error during Vertex AI Search retrieval: {e}")
            return None
    
    def _rag_corpus_resoruce(self) -> str:
        """Return fully qualified name of Rag corpus."""
        return f"projects/{self.project_id}/locations/{self.location}/ragCorpora/{self.rag_corpus_id}"

    def _rag_retrieval_config(self) -> rag.RagRetrievalConfig:
        return rag.RagRetrievalConfig(top_k=3, filter=rag.Filter(vector_distance_threshold=0.5))
        



