from pydantic import BaseModel
from typing import List
from models.retrieved_context import RetrievedContext

class LogEntry(BaseModel):
    user_query: str
    retrieved_contexts: List[RetrievedContext]
    llm_output: str
    timestamp: str
    latency: float
    rewritten_query: str | None
    rewrite_strategy: str #TODO - change to enumerated type
    