from pre_retrieval.query_optimisation.base import QueryRewriter
from services.gemini_service import GeminiService

class MultiQueryStrategy(QueryRewriter):
    """Performs multiqueryexpansion - simplifying and broadening the query."""

    def __init__(self, gemini_service: GeminiService):
        self.service = gemini_service
    
    def rewrite(self, query: str) -> list[str]:
        rewritten = self.service.rewrite_query(query)
        return rewritten