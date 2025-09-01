from fastapi import FastAPI, Depends
from logging_modules.logging_config import setup_logging 
from models.user_query import UserQuery
from services.retrieval_service import single_pass
from dependencies import get_gemini_client, get_vertex_retrieval_client
from clients.vertex_retrieval_client import VertexRetrievalClient
from clients.gemini_client import GeminiClient

app = FastAPI()
setup_logging()

@app.post("/query")
async def service_query(
    query: UserQuery,
    retriever: VertexRetrievalClient = Depends(get_vertex_retrieval_client),
    llm_client: GeminiClient = Depends(get_gemini_client)
):
    text = await single_pass(query.text, retriever, llm_client)
    return {"text": text}
