from fastapi import FastAPI, Depends
from logging_modules.logging_config import setup_logging 
from models.user_query import UserQuery
from dependencies import get_gemini_client, get_retriever_router, get_answer_critic
from agentic.retrieval_router import RetrievalRouter 
from agentic.answer_critic import AnswerCritic
from services.retrieval_service import run
from clients.gemini_client import GeminiClient

MAX_ITERATIVE_STEPS = 2

app = FastAPI()
setup_logging()

@app.post("/query")
async def service_query(
    query: UserQuery,
    retriever_router: RetrievalRouter = Depends(get_retriever_router),
    llm_client: GeminiClient = Depends(get_gemini_client),
    answer_critic: AnswerCritic = Depends(get_answer_critic) 
):
    retriever = retriever_router.get_retriever(query.text)
    text = await run(query.text, retriever, llm_client, answer_critic, MAX_ITERATIVE_STEPS)
    return {"text": text}
