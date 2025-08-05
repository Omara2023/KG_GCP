import os
from fastapi import Depends
from clients.gemini_client import GeminiClient
from llm_behaviours.base import QueryRewriter
from llm_behaviours.identity import IdentityRewriter
from llm_behaviours.step_back import StepBackRewriter
from llm_behaviours.multi_query import MultiQueryExpander
from llm_behaviours.grounded_generator import GroundedAnswerGenerator
from services.gemini_service import GeminiService

from clients.biq_query_client import BigQueryClient
from services.big_query_service import BigQueryService

#Gemini factories:

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

#Vertex factories:

#Big Query Factories:

def get_big_query_client() -> BigQueryClient:
    project_id=os.getenv("GCP_PROJECT_ID")
    dataset_id=os.getenv("GCP_BQ_DATASET_ID")
    table_id=os.getenv("GCP_BQ_TABLE_ID")
    if project_id is None or dataset_id is None or table_id is None:
        raise ValueError("Cannot instantiate BigQuery connection with missing env.")
    
    return BigQueryClient(project_id, dataset_id, table_id)

def get_big_query_service(client: BigQueryClient = Depends(get_big_query_client)) -> BigQueryService:
    return BigQueryService(client)