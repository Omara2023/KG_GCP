import json
import logging
from llm_behaviours.pre_retrieval.base import QueryRewriter
from clients.gemini_client import GeminiClient

class SubQueryExpander(QueryRewriter):
    """Class responsible for generating decomposed parts of the original query."""  
    
    def __init__(self, llm_client: GeminiClient):
        self.llm = llm_client
        self.logger = logging.getLogger(__name__)

    def rewrite(self, query: str, n = 5) -> list[str]:
        prompt = (
            f"Break down the following query into {n} sub-queries targeting different aspects of the query: {query}.\n"
            f"Respond only with a valid JSON array of objects with a single query key. No explanations."
        )
        output = self.llm.prompt(prompt)
        return self._safe_parse_json(output) 
    
    def _safe_parse_json(self, text: str) -> list[str]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find('[')
            end = text.find(']')
            if start != -1 and end != -1 and start < end:
                try:
                    data = json.loads(text[start:end+1])
                except json.JSONDecodeError:
                    raise ValueError("Could not parse JSON from response.")
            else:
                raise ValueError("Could not parse JSON from response.")

        if isinstance(data, list) and all(isinstance(item, dict) and 'query' in item for item in data):
            return [item["query"] for item in data]
        else:
            raise ValueError("Parsed JSON structure unexpected.")