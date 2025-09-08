import json
from llm_behaviours.pre_retrieval.base import QueryRewriter

class MultipleOutput(QueryRewriter):
    """ABC for rewriters that produce multiple output queries."""
    
    def _safe_parse_json_list(self, text: str) -> list[str]:
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