import json
from clients.gemini_client import GeminiClient

class MultiQueryExpander:
    """Class responsible for generate variations of the original query."""  
    def __init__(self, llm_client: GeminiClient):
        self.llm = llm_client

    def rewrite(self, query: str, n = 5) -> list[str]:
        prompt = (
            f"Generate {n} variations of the following query:{query}:\n"
            f"Respond only with a valid JSON array of objects with a single query key. No explanations."
        )
        output = self.llm.prompt(prompt)
        return self._safe_parse_json(output) 
    
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
