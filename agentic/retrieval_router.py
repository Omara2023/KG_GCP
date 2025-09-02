from json import loads
from typing import Any
from clients.gemini_client import GeminiClient
from clients.vertex_retrieval_client import VertexRetrievalClient
#future types: simple, hydrid, keyword???, answer_in_question...

class RetrievalRouter:
    """Decide which retrieval strategy should be used for a query."""
    def __init__(self, retrievers: dict[str, Any], llm_client: GeminiClient):
        self.retrievers = retrievers
        self.llm = llm_client
        self.llm.system_instruction = "You are an expert decision maker, deciding what type of RAG lookup strategy to use in order to answer user queries factually and usefully."
        
    def get_retriever(self, query: str) -> VertexRetrievalClient: #refactor to have a family of retriever classes implementing a retiver interface
        prompt = self._contruct_prompt(query)
        output = self.llm.prompt(prompt)
        option = loads(output)["choice"]
        if option not in self.retrievers:
            raise ValueError("Incorrect retriever type returned by LLM.")
        return self.retrievers[option]
    
    def _contruct_prompt(self, query: str) -> str:
        return (
            "Decide which retrieval strategy is best suited for a RAG based system to answer the user query.\n"
            "Types:\n"
            f"{', '.join(f'{i}' for i in self.retrievers.keys())}\n"
            f"User query: {query}\n"
            "Return your answer as JSON with key 'choice' and value <TYPE>"
        )
