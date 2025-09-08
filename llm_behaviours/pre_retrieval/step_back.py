from llm_behaviours.pre_retrieval.base import QueryRewriter
from clients.gemini_client import GeminiClient

class StepBackRewriter(QueryRewriter):
    """Class responsible for rewriting queries via step-back logic."""  

    def __init__(self, llm_client: GeminiClient):
        self.llm = llm_client

    def __str__(self) -> str:
        return "Rewrite the query into a more general/informative form."

    def rewrite(self, query: str) -> list[str]:
        prompt = self._prompt(query)
        return [self.llm.prompt(prompt)]
    
    def _prompt(self, query: str) -> str:
        return (
            "You are a helpful assistant.\n"
            f"Rewrite the following user query into a more general or informative form for knowledge retrieval:\n\n"
            f"Original query: {query}"
        )