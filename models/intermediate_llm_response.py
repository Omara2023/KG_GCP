from pydantic import BaseModel

class IntermediateLLMResponse(BaseModel):
    """Class to carry reponses from the 1 to n-1 responses of the llm in iterative generation."""

    intermediate_answer: str
    next_query: str