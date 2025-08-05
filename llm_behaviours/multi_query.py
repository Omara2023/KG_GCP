import json
import logging
from llm_behaviours.base import QueryRewriter
from clients.gemini_client import GeminiClient

class MultiQueryExpander(QueryRewriter):
    """Class responsible for generate variations of the original query."""  
    
    def __init__(self, llm_client: GeminiClient):
        self.llm = llm_client
        self.logger = logging.getLogger(__name__)

    def rewrite(self, query: str, n = 5) -> list[str]:
        prompt = (
            f"Generate {n} variations of the following query:{query}:\n"
            f"Respond only with a valid JSON array of objects with a single query key. No explanations."
        )
        output = self.llm.prompt(prompt)
        self.logger.info(f"Type before json parsing: {type(output)}")
        self.logger.info(f"Actual output: {output}")

        to_return = self._safe_parse_json(output) 
        self.logger.info(f"Type post json parsing: {type(to_return)}")
        self.logger.info(f"Actual output: {to_return}")

        return to_return
    
    def _safe_parse_json(self, text: str) -> list[str]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find('[')
            end = text.find(']')
            if start != -1 and end != -1 and start < end:
                try:
                    return json.loads(text[start:end+1])    
                except json.JSONDecodeError:
                    pass
            raise ValueError("Could not parse JSON from response.")
