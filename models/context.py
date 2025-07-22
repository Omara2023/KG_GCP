from dataclasses import dataclass
from typing import Any

@dataclass
class Context:
    text: str
    source_uri: str
    score: float

    @classmethod
    def from_proto(cls, proto_ctx: Any) -> "Context":
        return cls(
            text=proto_ctx.text,
            source_uri=proto_ctx.source_uri,
            score = proto_ctx.score,
        )