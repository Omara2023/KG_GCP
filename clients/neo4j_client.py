import logging
from neo4j import Driver

class Neo4jClient:
    """Encapsulates Neo4j driver logic and transactions."""

    def __init__(self, driver: Driver):
        """GraphDatabase driver to be passed in."""
        self.driver = driver        
        self.logger = logging.getLogger(__name__)

    def create_vector_index(self) -> None:
        query = """CREATE VECTOR INDEX pdf_chunks IF NOT EXISTS
            FOR (c:Chunk)
            ON c.embedding
        """
        self.driver.execute_query(query)

    def create_constraints(self) -> None:
        """Create all necessary constraints for PDFs and Chunk nodes."""
        queries = [
            "CREATE CONSTRAINT pdf_id_unique IF NOT EXISTS FOR (p:PDF) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT pdf_title_unique IF NOT EXISTS FOR (p:PDF) REQUIRE p.title IS UNIQUE",
            "CREATE CONSTRAINT chunk_unique_per_pdf IF NOT EXISTS FOR (c:Chunk) REQUIRE (c.pdf_id, c.index) IS NODE KEY",
            "CREATE VECTOR INDEX pdf_chunks IF NOT EXISTS FOR (c:Chunk) ON c.embedding"
        ]
        for query in queries:
            self.driver.execute_query(query) # type: ignore

    def store_chunks(self, pdf_id: str, pdf_title: str, chunks: list[dict]):
        """
        'chunks' should be a lst of dicts built like so:
        [{"text": "...", "embedding": [...]}, ...]
        """
        query = """
            MERGE (p:PDF {id: $pdf_id})
            ON CREATE SET p.title = $title, p.created_at = datetime() 
            SET p.updated_at = datetime()

            WITH p
            UNWIND $chunks AS chunk 
            MERGE (c:Chunk {pdf_id: $pdf_id, index: chunk.index}) 
            SET c.text = chunk.text, c.embedding = chunk.embedding
            MERGE (p)-[:HAS_CHUNK]->(c)  
        """
        self.driver.execute_query(query, pdf_id=pdf_id, title=pdf_title, chunks=chunks)

    def knn_search(self, pdf_id: str | None, query_embedding: list[float], n: int = 5) -> list:
        base_query = """
            WITH $query_embedding AS q
            MATCH (c:Chunk)
            {where_clause}
            WITH c, gds.similarity.cosine(c.embedding, q) AS score
            RETURN c, score
            ORDER BY score DESC
            LIMIT $n    
        """

        where_clause = "WHERE c.pdf = $pdf_id" if pdf_id else ""
        query = base_query.replace("{where_clause}", where_clause)

        params = {"query_embedding": query_embedding, "n": n}
        if pdf_id:
            params["pdf_id"] = pdf_id    
            
        result, _, _ = self.driver.execute_query(query, **params)
        return [(r["c.text"], r["score"]) for r in result]