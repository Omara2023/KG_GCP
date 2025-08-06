import os
import asyncio
from fastapi import FastAPI, Depends
from datetime import datetime
from dependencies import get_gemini_service, get_big_query_service, get_vertex_service_factory
from logging_modules.logging_config import setup_logging
from services.gemini_service import GeminiService
from services.big_query_service import BigQueryService
from models.user_query import UserQuery
from models.log_entry import LogEntry
from models.context import RetrievedContext
from models.llm_response import LLMResponse

app = FastAPI()

logger = setup_logging()

@app.post("/query", response_model=LLMResponse)
async def service_query(query: UserQuery, vertex_service_factory = Depends(get_vertex_service_factory) , gemini_service: GeminiService = Depends(get_gemini_service), big_query_service: BigQueryService = Depends(get_big_query_service)):
    rewrite_strategy = os.getenv("QUERY_REWRITE_STRATEGY", "identity") 
    start = datetime.now()

    logger.info(f"User query: '{query.text}'")
    rewritten_queries = gemini_service.rewrite_query(query.text)
    logger.info(f"{len(rewritten_queries)} rewritten queries derived.")

    async def fetch_contexts(q: str):
        service = vertex_service_factory()
        return await asyncio.to_thread(service.search, q)

    nested_results = await asyncio.gather(*(fetch_contexts(q) for q in rewritten_queries))
    contexts = [item for sublist in nested_results for item in sublist]
    
    logger.info(f"{len(contexts)} contexts produced in total.")
    contexts = list(set(contexts))
    logger.info(f"{len(contexts)} unique contexts.")
    response = _get_llm_response(gemini_service, query.text, contexts)
    _log_to_big_query(big_query_service, query.text, contexts, response, start, None, rewrite_strategy)
    
    return response    

def _get_llm_response(service: GeminiService, query: str, contexts: list[RetrievedContext]) -> LLMResponse:
    return LLMResponse(text=service.respond(query, contexts))

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
