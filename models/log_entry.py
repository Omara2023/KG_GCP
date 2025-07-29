from dataclasses import dataclass, asdict
from typing import List
from models.context import RetrievedContext

@dataclass
class LogEntry:
    user_query: str
    retrieved_contexts: List[RetrievedContext]
    llm_output: str
    timestamp: str
    latency: float
    
    def to_dict(self) -> dict:
        return asdict(self)