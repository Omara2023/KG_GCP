from llm_behaviours.base import QueryRewriter
from llm_behaviours.grounded_generator import GroundedAnswerGenerator 
from models.context import RetrievedContext

class GeminiService:
    """Orchestrates Gemini-powered capabilities."""
    
    def __init__(self, rewriter: QueryRewriter, generator: GroundedAnswerGenerator):
        self.rewriter = rewriter
        self.generator =  generator

    def rewrite_query(self, query: str) -> list[str]:
        return self.rewriter.rewrite(query)   
        
    def respond(self, query: str, contexts: list[RetrievedContext]) -> str:
        return self.generator.generate(query, contexts)
