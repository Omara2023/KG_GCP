from fastapi import FastAPI, Depends
from datetime import datetime
from logging_modules.logging_config import setup_logging
from services.vertex_service import VertexRagService, get_vertex_service
from services.gemini_service import GeminiService, get_gemini_service
from services.big_query_service import BigQueryService, get_big_query_service
from models.user_query import UserQuery
from models.log_entry import LogEntry
from models.context import RetrievedContext
from models.llm_response import LLMResponse

app = FastAPI()

logger = setup_logging()

@app.post("/query", response_model=LLMResponse)
async def service_query(query: UserQuery, vertex_service: VertexRagService = Depends(get_vertex_service), gemini_service: GeminiService = Depends(get_gemini_service), big_query_service: BigQueryService = Depends(get_big_query_service)):
    start = datetime.now()

    _log_query(query)
    contexts = _get_contexts(vertex_service, query)
    response = _get_llm_response(gemini_service, query, contexts)
    _log_to_big_query(big_query_service, query, contexts, response, start)
    
    return response

def _log_query(query: UserQuery) -> None:
    logger.info(f"User query: '{query.text}'")

def _get_contexts(service: VertexRagService, query: UserQuery) -> list[RetrievedContext]:
    contexts = service.search(query.text)
    if contexts:
        logger.info("Successfully retrieved contexts from RAG engine.")
    else:
        logger.info("Failed to retrieve contexts from RAG engine.")  
    return contexts

def _get_llm_response(service: GeminiService, query: UserQuery, contexts: list[RetrievedContext]) -> LLMResponse:
    return service.respond(query.text, contexts)

def _log_to_big_query(service: BigQueryService, query: UserQuery, contexts: list[RetrievedContext], response: LLMResponse, start_time: datetime) -> None:
    elapsed = (datetime.now()  - start_time).total_seconds()
    log_entry = LogEntry(user_query=query.text, 
                         retrieved_contexts=contexts,
                         llm_output=response.text, 
                         timestamp=datetime.now().isoformat(), 
                         latency=elapsed)
    service.log_query(log_entry)
