import os
from fastapi import Depends
from pre_retrieval.query_optimisation.base import QueryRewriter
from pre_retrieval.query_optimisation.identity import IndentityStrategy
from pre_retrieval.query_optimisation.step_back import StepBackStrategy
from services.gemini_service import GeminiService, get_gemini_service

def get_query_rewriter(gemini_service: GeminiService = Depends(get_gemini_service)) -> QueryRewriter:
    strategy = os.getenv("QUERY_REWRITE_STRATEGY", "identity")
    
    if strategy == "identity":
        return IndentityStrategy()
    elif strategy == "step_back":
        return StepBackStrategy(gemini_service)
    else:
        raise ValueError(f"Unknown query rewrite strategy: {strategy}") 
