from typing import Optional, Any
from models.context import Context

class RetrievedContext(Context):
    """Represents a chunk retrieved from the RAG-corpus."""
    text: str
    source_file: Optional[str] = None
    score: Optional[float] = None

    @classmethod
    def from_proto(cls, proto_ctx: Any) -> "RetrievedContext":
        return cls(
            text=proto_ctx.text,
            source_file=proto_ctx.source_uri,
            score = proto_ctx.score,
        )
    
    def to_user_friendly(self) -> str:
        return f"{self.source_file} - {self.score} - {self.text}"
    
    def __eq__(self, other):
        if isinstance(other, RetrievedContext):
            return (self.text, self.source_file) == (other.text, other.source_file)
        return False 
    
    def __hash__(self):
        return hash((self.text, self.source_file))
    