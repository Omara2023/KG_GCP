import os
from fastapi import Depends
from neo4j import Driver, GraphDatabase
from clients.gemini_client import GeminiClient
from clients.vertex_client import VertexClient
from clients.neo4j_client import Neo4jClient
from indexing_pipeline.embedder import Embedder
from llm_behaviours.pre_retrieval.base import QueryRewriter
from llm_behaviours.pre_retrieval.identity import IdentityRewriter
from llm_behaviours.pre_retrieval.step_back import StepBackRewriter
from llm_behaviours.pre_retrieval.multi_query import MultiQueryExpander
from llm_behaviours.pre_retrieval.sub_query import SubQueryExpander
from retrievers.vertex_retriever import VertexRetriever
from retrievers.graph_retriever import GraphRetriever
from agentic.retrieval_router import RetrievalRouter
from agentic.answer_critic import AnswerCritic
from agentic.query_rewrite_router import QueryRewriteRouter

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

#Vertex factories:

def get_vertex_client() -> VertexClient:
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_RAG_CORPUS_REGION")
    rag_corpus_id = os.getenv("GCP_RAG_CORPUS_ID")

    if project_id is None or location is None or rag_corpus_id is None:
        raise ValueError("Cannot instantiate vertexai connection with missing env.")
    return VertexClient(project_id, location, rag_corpus_id)

def get_vertex_retriever(client: VertexClient = Depends(get_vertex_client)) -> VertexRetriever:
    return VertexRetriever(client)

#Neo4j factories:

def get_neo4j_driver() -> Driver:
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if uri is None or username is None or password is None:
        exit(1)

    return GraphDatabase.driver(uri, auth=(username, password))

def get_neo4j_client(driver: Driver = Depends(get_neo4j_driver)) -> Neo4jClient:
    return Neo4jClient(driver)

#Graph DB factories:

def get_graph_retriever(client: Neo4jClient = Depends(get_neo4j_client)) -> GraphRetriever:
    embedder = Embedder() #change this to use DI and extend to use ENV for different embedding types project wide.
    return GraphRetriever(embedder, client)

#Rewriter router factory:

def get_rewriter_router(llm_client: GeminiClient = Depends(get_gemini_client)) -> QueryRewriteRouter:    
    rewriters = {
        "identity": IdentityRewriter(),
        "step_back": StepBackRewriter(get_gemini_client()),
        "multi_query": MultiQueryExpander(get_gemini_client()),
        "sub_query": SubQueryExpander(get_gemini_client())
    }

    return QueryRewriteRouter(rewriters, llm_client)


#Retriever router factory:

def get_retriever_router(vertex: VertexRetriever = Depends(get_vertex_retriever), 
                         graph: GraphRetriever = Depends(get_graph_retriever),
                         llm_client: GeminiClient = Depends(get_gemini_client)) -> RetrievalRouter:
    retrievers = {"graph": graph} # "vertex": vertex, add that back in once vertex is re-enabled.
    return RetrievalRouter(retrievers, llm_client) 

#Answer critic factories:

def get_answer_critic(llm_client: GeminiClient = Depends(get_gemini_client)) -> AnswerCritic:
    return AnswerCritic(llm_client)

#Big Query Factories:

# def get_big_query_client() -> BigQueryClient:
#     project_id=os.getenv("GCP_PROJECT_ID")
#     dataset_id=os.getenv("GCP_BQ_DATASET_ID")
#     table_id=os.getenv("GCP_BQ_TABLE_ID")

#     if project_id is None or dataset_id is None or table_id is None:
#         raise ValueError("Cannot instantiate BigQuery connection with missing env.")
#     return BigQueryClient(project_id, dataset_id, table_id)

