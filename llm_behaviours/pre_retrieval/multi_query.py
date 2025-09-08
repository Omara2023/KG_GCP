import logging
from llm_behaviours.pre_retrieval.multiple import MultipleOutput
from clients.gemini_client import GeminiClient

class MultiQueryExpander(MultipleOutput):
    """Class responsible for generate variations of the original query."""  

    DESCRIPTION = "Generate n variations of the query."
    
    def __init__(self, llm_client: GeminiClient):
        self.llm = llm_client
        self.logger = logging.getLogger(__name__)

    def rewrite(self, query: str, n = 5) -> list[str]:
        prompt = self._prompt(query, n)
        output = self.llm.prompt(prompt)
        return self._safe_parse_json_list(output) 
    
    def _prompt(self, query: str, n: int) -> str:
        return (
            f"Generate {n} variations of the following query:{query}:\n"
            f"Respond only with a valid JSON array of objects with a single query key. No explanations."
        )
