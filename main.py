from fastapi import FastAPI, Depends
from pydantic import BaseModel
from datetime import datetime
from logging_modules.logging_config import setup_logging

from services.vertex_service import VertexRagService
from services.gemini_service import GeminiService
from services.big_query_service import BigQueryService
from models.log_entry import LogEntry

class UserQuery(BaseModel):
    text: str

app = FastAPI()

logger = setup_logging()

@app.post("/query")
async def service_query(query: UserQuery,
                        vertex_service: VertexRagService = Depends):
    start = datetime.now()

    logger.info(f"User query: '{query.text}'")

    vertex_service = VertexRagService() 
    if (contexts := vertex_service.search(query.text)):
        logger.info("Successfully retrieved contexts from RAG engine.")
    else:
        logger.info("Failed to retrieve contexts from RAG engine.")     
    
    gemini_service = GeminiService()
    response = gemini_service.respond(query.text, contexts)

    big_query_service = BigQueryService()
    elapsed = (datetime.now()  - start).total_seconds()
    log_entry = LogEntry(query.text, contexts, response["response"], datetime.now().isoformat(), elapsed)
    big_query_service.log_query(log_entry)

    return response



