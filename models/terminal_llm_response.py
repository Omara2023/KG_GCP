from models.llm_response import LLMResponse

class TerminalLLMResponse(LLMResponse):
    """Final responses from LLM to be returned to user."""
    text: str

    def to_user_friendly(self) -> str:
        return self.text