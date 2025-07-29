from dataclasses import dataclass, asdict
from typing import List
from models.context import RetrievedContext


@dataclass
class LogEntry:
    user_query: str
    retrieved_contexts: List[RetrievedContext]
    output: str
    latency: float
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)