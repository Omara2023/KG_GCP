from models.llm_response import LLMResponse

class TerminalLLMResponse(LLMResponse):
    """Final responses from LLM to be returned to user."""
    text: str
