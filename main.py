from fastapi import FastAPI, Depends, Request
from cloudevents.http import from_http
from logging_modules.logging_config import setup_logging 
from models.user_query import UserQuery
from dependencies import get_gemini_client, get_retriever_router, get_answer_critic, get_rewriter_router
from agentic.retrieval_router import RetrievalRouter 
from agentic.answer_critic import AnswerCritic
from agentic.query_rewrite_router import QueryRewriteRouter
from services.retrieval_service import run
from clients.gemini_client import GeminiClient

MAX_ITERATIVE_STEPS = 2

app = FastAPI()
logger = setup_logging()

@app.post("/query")
async def service_query(
    query: UserQuery,
    rewriter_router: QueryRewriteRouter = Depends(get_rewriter_router),
    retriever_router: RetrievalRouter = Depends(get_retriever_router),
    llm_client: GeminiClient = Depends(get_gemini_client),
    answer_critic: AnswerCritic = Depends(get_answer_critic) 
):
    rewriter = rewriter_router.select_rewriter(query.text)
    retriever = retriever_router.get_retriever(query.text)
    text = await run(query.text, rewriter, retriever, llm_client, answer_critic, MAX_ITERATIVE_STEPS)
    return {"text": text}

@app.post("/index")
async def receieve_event(request: Request):
    event = from_http(request.headers, await request.body())
    logger.info(f"Got event {event['type']} for {event.data['bucket']}/{event.data['name']}")
    logger.info(f"event: {event}")
    logger.info(f"dir(event): {dir(event)}")
    return {"ok": True}