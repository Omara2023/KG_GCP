import os
from fastapi import Depends
from clients.gemini_client import GeminiClient
from llm_behaviours.base import QueryRewriter
from llm_behaviours.identity import IdentityRewriter
from llm_behaviours.step_back import StepBackRewriter
from llm_behaviours.multi_query import MultiQueryExpander
from llm_behaviours.grounded_generator import GroundedAnswerGenerator
from services.gemini_service import GeminiService

def get_gemini_client() -> GeminiClient:
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_GEMINI_REGION")
    return GeminiClient(project_id, location)

def get_query_rewriter(llm_client: GeminiClient = Depends(get_gemini_client)) -> QueryRewriter:
    strategy = os.getenv("QUERY_REWRITE_STRATEGY")
    
    match strategy:
        case "identity":
            return IdentityRewriter()
        case "step_back":
            return StepBackRewriter(llm_client)
        case "multi_query":
            return MultiQueryExpander(llm_client)
        case _:
            raise ValueError(f"Unknown query rewrite strategy: {strategy}") 

def get_answer_generator(llm_client: GeminiClient = Depends(get_gemini_client)) -> GroundedAnswerGenerator:
    return GroundedAnswerGenerator(llm_client)

def get_gemini_service(rewriter: QueryRewriter = Depends(get_query_rewriter), generator: GroundedAnswerGenerator = Depends(get_answer_generator)) -> GeminiService:
    return GeminiService(rewriter, generator)