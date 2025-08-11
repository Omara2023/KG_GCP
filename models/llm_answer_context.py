from models.context import Context

class LLMAnswerContext(Context):
    """Context representation of an intermediate answer from iterative generation."""
    text: str

    def to_user_friendly(self) -> str:
        return f"LLM generated context - {self.text}"