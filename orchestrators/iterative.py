import asyncio
import logging
from typing import Callable
from models.llm_response import LLMResponse
from models.intermediate_llm_response import IntermediateLLMResponse
from models.llm_answer_context import LLMAnswerContext
from orchestrators.base import BaseRAGOrchestrator
from llm_behaviours.pre_retrieval.base import QueryRewriter
from llm_behaviours.generation.iterative_step import IterativeStepGenerator
from llm_behaviours.generation.final_answer_aggregator import FinalAnswerAggregator
from clients.vertex_retrieval_client import VertexRetrievalClient

class IterativeRAGOrchestrator(BaseRAGOrchestrator):
    """Iterative pipeline of generate, re-retrieve n times."""

    def __init__(self, query_rewriter: QueryRewriter, retriever_factory: Callable[[], VertexRetrievalClient], iterative_step_generator: IterativeStepGenerator, final_answer_generator: FinalAnswerAggregator, n: int = 3) -> None:
        if n <= 1: raise ValueError(f"Iterative generation cannot be done {n} times.")
        self.query_rewriter = query_rewriter
        self.retriever_factory = retriever_factory
        self.iterative_step_generator = iterative_step_generator
        self.final_answer_generator = final_answer_generator
        self.n = n
        self.logger = logging.getLogger(__name__)

    async def generate(self, query: str) -> LLMResponse:
        self.logger.info(f"Original query: {query}")

        self.logger.info("Rewritten queries.")
        rewritten_queries = self.query_rewriter.rewrite(query)
        for q in rewritten_queries:
            self.logger.info(q)

        answers: list[LLMAnswerContext] = []
        nested_results = await asyncio.gather(*(self._fetch_contexts(q) for q in rewritten_queries))
        contexts = [item for sublist in nested_results for item in sublist]
        contexts = list(set(contexts))
        response = self.iterative_step_generator.generate(query, contexts)
        
        if (not isinstance(response, IntermediateLLMResponse)):
            raise ValueError(f"Incorrect return type {type(response)}. Expected IntermediateLLMResponse.")
        
        prompt = response.next_query
        answers.append(LLMAnswerContext(text=response.intermediate_answer))

        self.logger.info(f"Query 1: {prompt}")

        for i in range(1, self.n - 1):
            contexts = self.retriever_factory().run_context_retrieval(prompt)
            response = self.iterative_step_generator.generate(prompt, contexts)

            if (not isinstance(response, IntermediateLLMResponse)):
                raise ValueError(f"Incorrect return type {type(response)}. Expected IntermediateLLMResponse.")
            
            prompt = response.next_query
            answers.append(LLMAnswerContext(text=response.intermediate_answer))
            self.logger.info(f"Query {i + 1}: {prompt}")

        self.logger.info(f"Pre final answer length of answers list: {len(answers)}")
        self.logger.info("Answers list:")
        for a in answers:
            self.logger.info(a)
        response = self.final_answer_generator.generate(query, answers)

        return response

    async def _fetch_contexts(self, query: str):
        service = self.retriever_factory()
        return await asyncio.to_thread(service.run_context_retrieval, query)

