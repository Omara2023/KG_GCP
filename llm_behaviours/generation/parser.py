import json

def _safe_parse_json(text: str) -> str:
    """Parse out desired JSON object from LLM response."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.find('}')
        if start != -1 and end != -1 and start < end:
            try:
                data = json.loads(text[start:end+1])
            except json.JSONDecodeError:
                raise ValueError("Could not parse JSON from response.")
        else:
            raise ValueError("Could not parse JSON from response.")
        
    return data