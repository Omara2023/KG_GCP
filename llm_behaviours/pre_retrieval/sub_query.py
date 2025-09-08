import logging
from llm_behaviours.pre_retrieval.multiple import MultipleOutput
from clients.gemini_client import GeminiClient

class SubQueryExpander(MultipleOutput):
    """Class responsible for generating decomposed parts of the original query."""  
    
    DESCRIPTION = "Break down the query into n sub-queries targeting different aspects of the query."

    def __init__(self, llm_client: GeminiClient):
        self.llm = llm_client
        self.logger = logging.getLogger(__name__)

    def rewrite(self, query: str, n = 5) -> list[str]:
        prompt = self._prompt(query, n)
        output = self.llm.prompt(prompt)
        return self._safe_parse_json_list(output) 
    
    def _prompt(self, query: str, n: int) -> str:
        return (
            f"Break down the following query into {n} sub-queries targeting different aspects of the query: {query}.\n"
            f"Respond only with a valid JSON array of objects with a single query key. No explanations."
        )

    