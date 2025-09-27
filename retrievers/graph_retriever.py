import logging
from indexing_pipeline.embedder import Embedder
from models.retrieved_context import RetrievedContext
from retrievers.base import Retriever
from clients.neo4j_client import Neo4jClient

class GraphRetriever(Retriever):
    """Application-level wrapper that adapts GraphDB search results."""

    def __init__(self, embedder: Embedder, neo4j_client: Neo4jClient):
        self.embedder = embedder
        self.client = neo4j_client
        self.logger = logging.getLogger(__name__)

    def run_context_retrieval(self, query: str) -> list[RetrievedContext]: #TODO - find a way to include pdf_idf in interface overload while not breaking vertex "id-less" version.
        query_embedding = self.embedder.embed_query(query) 
        result = self.client.knn_search(None, query_embedding)
        return [RetrievedContext(text=r[0]) for r in result] #TODO - consider using score attribute in result tuple. Atm we merely return the Neo4j chunk.text
