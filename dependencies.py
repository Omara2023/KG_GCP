import os
from typing import Callable
from fastapi import Depends
from clients.gemini_client import GeminiClient
from clients.vertex_retrieval_client import VertexRetrievalClient
from llm_behaviours.pre_retrieval.base import QueryRewriter
from llm_behaviours.pre_retrieval.identity import IdentityRewriter
from llm_behaviours.pre_retrieval.step_back import StepBackRewriter
from llm_behaviours.pre_retrieval.multi_query import MultiQueryExpander
from llm_behaviours.pre_retrieval.sub_query import SubQueryExpander
from llm_behaviours.generation.base import AnswerGenerator
from llm_behaviours.generation.grounded import GroundedAnswerGenerator
from orchestrators.base import BaseRAGOrchestrator
from orchestrators.single_pass import SinglePassRAGOrchestrator

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
        case "sub_query":
            return SubQueryExpander(llm_client)
        case _:
            raise ValueError(f"Unknown query rewrite strategy: {strategy}") 

def get_grounded_answer_generator(llm_client: GeminiClient = Depends(get_gemini_client)) -> GroundedAnswerGenerator:
    return GroundedAnswerGenerator(llm_client)

#Vertex factories:

def get_vertex_retrieval_client() -> VertexRetrievalClient:
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_RAG_CORPUS_REGION")
    rag_corpus_id = os.getenv("GCP_RAG_CORPUS_ID")

    if project_id is None or location is None or rag_corpus_id is None:
        raise ValueError("Cannot instantiate vertexai connection with missing env.")
    return VertexRetrievalClient(project_id, location, rag_corpus_id)


#Orchestrator factories:

def get_single_pass_rag_orchestrator(rewriter: QueryRewriter = Depends(get_query_rewriter), 
                                     retriever_factory: Callable[[], VertexRetrievalClient] = get_vertex_retrieval_client, 
                                     generator: AnswerGenerator = Depends(get_grounded_answer_generator)) -> SinglePassRAGOrchestrator:
    return SinglePassRAGOrchestrator(rewriter, retriever_factory, generator)

def get_iterative_rag_orchestrator():
    pass

def get_orchestrator(single_pass: SinglePassRAGOrchestrator = Depends(get_single_pass_rag_orchestrator)) -> BaseRAGOrchestrator:
    retrieval_strategy = os.getenv("RETRIEVAL_STRATEGY")
    
    match retrieval_strategy:
        case "single_pass":
            return single_pass
        # case "iterative":
        #     iterative
        case _:
            raise ValueError(f"Unknown retrieval strategy: {retrieval_strategy}") 

#Big Query Factories:

# def get_big_query_client() -> BigQueryClient:
#     project_id=os.getenv("GCP_PROJECT_ID")
#     dataset_id=os.getenv("GCP_BQ_DATASET_ID")
#     table_id=os.getenv("GCP_BQ_TABLE_ID")

#     if project_id is None or dataset_id is None or table_id is None:
#         raise ValueError("Cannot instantiate BigQuery connection with missing env.")
#     return BigQueryClient(project_id, dataset_id, table_id)

