from dataclasses import dataclass
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
        return {
            "user_query": self.user_query,
            "retrieved_contexts": [ctx.to_dict() for ctx in self.retrieved_contexts],
            "llm_output": self.llm_output,
            "timestamp": self.timestamp,
            "latency": self.latency             
        }