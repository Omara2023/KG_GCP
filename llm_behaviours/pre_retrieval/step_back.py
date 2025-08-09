from llm_behaviours.pre_retrieval.base import QueryRewriter
from clients.gemini_client import GeminiClient

class StepBackRewriter(QueryRewriter):
    """Class responsible for rewriting queries via step-back logic."""  
    
    def __init__(self, llm_client: GeminiClient):
        self.llm = llm_client

    def rewrite(self, query: str) -> list[str]:
        prompt = (
            "You are a helpful assistant.\n"
            f"Rewrite the following user query into a more general or informative form for knowledge retrieval:\n\n"
            f"Original query: {query}"
        )
        return [self.llm.prompt(prompt)]