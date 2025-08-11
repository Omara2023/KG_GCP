import asyncio
import time
from functools import wraps
from typing import Callable
from services.vertex_service import VertexRagService
from services.gemini_service import GeminiService
from models.terminal_llm_response import TerminalLLMResponse

def log_duration(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        #log request in new bq table
        return result
    return wrapper

async def single_pass(query: str, vertex_service_factory: Callable[[], VertexRagService], gemini_service: GeminiService) -> TerminalLLMResponse:
    """Single-pass RAG implementation: query -> rewrite -> retrieve -> generate."""
    rewritten_queries = gemini_service.rewrite_query(query)

    async def fetch_contexts(q: str):
        service = vertex_service_factory()
        return await asyncio.to_thread(service.search, q)

    nested_results = await asyncio.gather(*(fetch_contexts(q) for q in rewritten_queries))
    contexts = list({item for sublist in nested_results for item in sublist})

    output = gemini_service.respond(query, contexts)

    if (isinstance(output, TerminalLLMResponse)):
        return output
    raise ValueError(f"Incorrect return type: {type(output)} instead of expected: {type(TerminalLLMResponse(text=''))}")
    


# def _log_to_big_query(service: BigQueryService, query: str, contexts: list[RetrievedContext], response: LLMResponse, start_time: datetime, rewritten_query: str|None, rewrite_strategy: str) -> None:
#     elapsed = (datetime.now()  - start_time).total_seconds()
#     log_entry = LogEntry(user_query=query, 
#                          retrieved_contexts=contexts,
#                          llm_output=response.text, 
#                          timestamp=datetime.now().isoformat(), 
#                          latency=elapsed,
#                          rewritten_query=rewritten_query,
#                          rewrite_strategy=rewrite_strategy)
#     service.log_query(log_entry)
