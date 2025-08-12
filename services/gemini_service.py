from typing import Sequence
from llm_behaviours.pre_retrieval.base import QueryRewriter
from llm_behaviours.generation.base import AnswerGenerator
from models.context import Context
from models.llm_response import LLMResponse

class GeminiService:
    """Orchestrates Gemini-powered capabilities."""
    
    def __init__(self, rewriter: QueryRewriter, generator: AnswerGenerator):
        self.rewriter = rewriter
        self.generator =  generator

    def rewrite_query(self, query: str) -> list[str]:
        return self.rewriter.rewrite(query)   
        
    def respond(self, query: str, contexts: Sequence[Context]) -> LLMResponse:
        return self.generator.generate(query, contexts)
