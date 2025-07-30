from pydantic import BaseModel
from typing import Optional, Any

class RetrievedContext(BaseModel):
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
    
    