from pre_retrieval.query_optimisation.base import QueryRewriter
from services.gemini_service import GeminiService

class StepBackStrategy(QueryRewriter):
    """Performs step-back rewriting - simplifying and broadening the query."""

    def __init__(self, gemini_service: GeminiService):
        self.rewriter = gemini_service.step_back_rewriter
    
    def rewrite(self, query: str) -> list[str]:
        rewritten = self.rewriter.rewrite(query)
        return [rewritten]