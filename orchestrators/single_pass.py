import asyncio
from typing import Callable
from models.llm_response import LLMResponse
from models.terminal_llm_response import TerminalLLMResponse
from orchestrators.base import BaseRAGOrchestrator
from llm_behaviours.pre_retrieval.base import QueryRewriter
from clients.vertex_retrieval_client import VertexRetrievalClient
from llm_behaviours.generation.base import AnswerGenerator

class SinglePassRAGOrchestrator(BaseRAGOrchestrator):
    """Baseline rewrite, semantic retrieval, generate pipeline."""

    def __init__(self, query_rewriter: QueryRewriter, retriever_factory: Callable[[], VertexRetrievalClient], generator: AnswerGenerator):
        self.query_rewriter = query_rewriter
        self.retriever_factory = retriever_factory
        self.generator = generator

    async def generate(self, query: str) -> LLMResponse:
        rewritten_queries = self.query_rewriter.rewrite(query)
        nested_results = await asyncio.gather(*(self._fetch_contexts(q) for q in rewritten_queries))
        contexts = list({item for sublist in nested_results for item in sublist})
        
        output = self.generator.generate(query, contexts)
            
        if (isinstance(output, TerminalLLMResponse)):
            return output
        raise ValueError(f"Incorrect return type: {type(output)} instead of expected: {type(TerminalLLMResponse(text=''))}")
    
    async def _fetch_contexts(self, query: str):
        service = self.retriever_factory()
        return await asyncio.to_thread(service.run_context_retrieval, query)

    


    

