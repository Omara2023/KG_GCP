import asyncio
import time
from functools import wraps
from typing import Callable
from services.vertex_service import VertexRagService
from services.gemini_service import GeminiService
from models.llm_response import LLMResponse
from models.intermediate_llm_response import IntermediateLLMResponse

def log_duration(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        #log request in new bq table
        return result
    return wrapper

async def iterative_retrieval(query: str, n: int, vertex_service_factory: Callable[[], VertexRagService], gemini_service: GeminiService) -> LLMResponse:
    """Iterative retrieval RAG: query -> (rewrite -> retrieve -> generate) * n -> return."""
    
    if n <= 1: raise ValueError(f"Iterative generation cannot be done {n} times.")
    
    rewritten_queries = gemini_service.rewrite_query(query)

    async def fetch_contexts(q: str):
        service = vertex_service_factory()
        return await asyncio.to_thread(service.search, q)
    
    answers = []
    for i in range(n):
        if i == 0:
            nested_results = await asyncio.gather(*(fetch_contexts(q) for q in rewritten_queries))
            contexts = [item for sublist in nested_results for item in sublist]
            contexts = list(set(contexts))
            response = gemini_service.respond(query, contexts)
            
            if (not isinstance(response, IntermediateLLMResponse)):
                raise ValueError(f"Incorrect return type {type(response)}. Expected IntermediateLLMResponse.")
            
            prompt = response.next_query
            answers.append(response.intermediate_answer)

        elif i < n - 1:
            contexts = vertex_service_factory().search(prompt)
            response = gemini_service.respond(prompt, contexts)

            if (not isinstance(response, IntermediateLLMResponse)):
                raise ValueError(f"Incorrect return type {type(response)}. Expected IntermediateLLMResponse.")
            
            prompt = response.next_query
            answers.append(response.intermediate_answer)

        else:

            response = gemini_service.respond(query, answers)

# def generate(self, query: str, contexts: list[RetrievedContext], n: int) -> list[str]:
#         """An"""
        
#         if not contexts: raise ValueError("No contexts given to grounded answer generator.")
        
        
#         prompt = query
#         answer_pool = []
#         for _ in range(n - 1):
#             output = self._ask_with_context(prompt, contexts)
#             response = self._safe_parse_json(output)
#             prompt = response.next_query
#             answer_pool.append(response.intermediate_answer)
#         return answer_pool

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
