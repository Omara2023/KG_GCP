from pydantic import BaseModel

class LLMResponse(BaseModel):
    """Abstract Class to model all responses from LLMs."""
    pass
