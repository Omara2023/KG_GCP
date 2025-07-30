from pydantic import BaseModel
from typing import List
from models.context import RetrievedContext

class LogEntry(BaseModel):
    user_query: str
    retrieved_contexts: List[RetrievedContext]
    llm_output: str
    timestamp: str
    latency: float
    