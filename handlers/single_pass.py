import asyncio
from datetime import datetime
from typing import Callable
from logging_modules.logging_config import setup_logging
from services.vertex_service import VertexRagService
from services.gemini_service import GeminiService
from services.big_query_service import BigQueryService
from models.log_entry import LogEntry
from models.context import RetrievedContext
from models.llm_response import LLMResponse


logger = setup_logging()

async def single_pass(query: str, vertex_service_factory: Callable[[], VertexRagService], rewrite_strategy: str, gemini_service: GeminiService, big_query_service: BigQueryService) -> LLMResponse:
    """Single-pass RAG implementation: query -> rewrite -> retrieve -> generate."""
    rewritten_queries = gemini_service.rewrite_query(query)

    async def fetch_contexts(q: str):
        service = vertex_service_factory()
        return await asyncio.to_thread(service.search, q)

    nested_results = await asyncio.gather(*(fetch_contexts(q) for q in rewritten_queries))
    contexts = [item for sublist in nested_results for item in sublist]
    contexts = list(set(contexts))
    
    response = LLMResponse(text=gemini_service.respond(query, contexts))
    return response    


def _log_to_big_query(service: BigQueryService, query: str, contexts: list[RetrievedContext], response: LLMResponse, start_time: datetime, rewritten_query: str|None, rewrite_strategy: str) -> None:
    elapsed = (datetime.now()  - start_time).total_seconds()
    log_entry = LogEntry(user_query=query, 
                         retrieved_contexts=contexts,
                         llm_output=response.text, 
                         timestamp=datetime.now().isoformat(), 
                         latency=elapsed,
                         rewritten_query=rewritten_query,
                         rewrite_strategy=rewrite_strategy)
    service.log_query(log_entry)
