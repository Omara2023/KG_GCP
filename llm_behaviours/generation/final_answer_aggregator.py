from typing import cast, Sequence
from clients.gemini_client import GeminiClient
from models.context import Context
from models.llm_answer_context import LLMAnswerContext 
from models.llm_response import LLMResponse
from models.terminal_llm_response import TerminalLLMResponse
from llm_behaviours.generation.base import AnswerGenerator

class FinalAnswerAggregator(AnswerGenerator):
    """Uses all of the intermediate answers and original query to prompt LLM."""  
    
    def __init__(self, llm_client: GeminiClient):
        self.llm = llm_client

    def generate(self, query: str, contexts: Sequence[Context]) -> LLMResponse:
        """Answer query using contexts, generating an intermediate answer and a subsequent question."""
        self._ensure_type(contexts, LLMAnswerContext)
        retrieved_contexts = cast(list[LLMAnswerContext], contexts)
        output = self._ask_with_context(query, retrieved_contexts)
        return TerminalLLMResponse(text=output)
        
    def _ask_with_context(self, user_query: str, contexts: Sequence[LLMAnswerContext]) -> str:
        """Construct prompt and invoke LLM."""
        prompt_parts = ["Using the following intermediate answers to answer the question.\n"]
        prompt_parts.append("--- Intermediaries ---")
        for i, context in enumerate(contexts):
            prompt_parts.append(f"\nContext {i+1}:\nText:{context.text}")
        prompt_parts.append("\n-------------------------\n")
        prompt_parts.append("Answer the following question as accurately as possible.")
        prompt_parts.append(f"\n\nQuestion: {user_query}")
        prompt = "\n".join(prompt_parts)
        
        return self.llm.prompt(prompt)
    
    