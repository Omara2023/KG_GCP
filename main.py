from os import getenv
from fastapi import FastAPI, Depends
from dependencies import get_gemini_service, get_vertex_service_factory
from handlers.single_pass import single_pass
from services.gemini_service import GeminiService
from models.user_query import UserQuery
from models.llm_response import LLMResponse

app = FastAPI()

@app.post("/query", response_model=LLMResponse)
async def service_query(query: UserQuery, vertex_service_factory = Depends(get_vertex_service_factory), gemini_service: GeminiService = Depends(get_gemini_service)):
    rewrite_strategy = getenv("QUERY_REWRITE_STRATEGY", "identity") 
    return await single_pass(query.text, vertex_service_factory, rewrite_strategy, gemini_service)
    
         

