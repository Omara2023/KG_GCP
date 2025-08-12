import json
from typing import NoReturn, cast, Sequence
from clients.gemini_client import GeminiClient
from models.context import Context
from models.retrieved_context import RetrievedContext
from llm_behaviours.generation.base import AnswerGenerator
from models.llm_response import LLMResponse
from models.intermediate_llm_response import IntermediateLLMResponse

class IterativeGenerator(AnswerGenerator):
    """Class responsible for the main augmented generation step."""  
    
    def __init__(self, llm_client: GeminiClient):
        self.llm = llm_client

    def generate(self, query: str, contexts: Sequence[Context]) -> LLMResponse:
        """Answer query using contexts, generating an intermediate answer and a subsequent question."""
        self._ensure_type(contexts, RetrievedContext)
        retrieved_contexts = cast(list[RetrievedContext], contexts)
        output = self._ask_with_context(query, retrieved_contexts)
        response = self._safe_parse_json(output)
        return response
        
    def _ask_with_context(self, user_query: str, contexts: Sequence[RetrievedContext]) -> str:
        """Construct prompt and invoke LLM."""
        prompt_parts = ["Using the following contexts answer the question.\n"]
        prompt_parts.append("--- Retrieved Contexts ---")
        for i, context in enumerate(contexts):
            prompt_parts.append(f"\nContext {i+1}:\nSource:{context.source_file}\nText:{context.text}")
        prompt_parts.append("\n-------------------------\n")
        prompt_parts.append("Answer the following question as accurately as possible and formulate a sensible subsequent query to be used for another iteration of rag-retrieval.")
        prompt_parts.append("Your response should be formatted as json, with the keys: \"next_query\" and \"intermediate_answer\" respectively.")
        prompt_parts.append(f"\n\nQuestion: {user_query}")
        prompt = "\n".join(prompt_parts)
        
        return self.llm.prompt(prompt)
    
    def _safe_parse_json(self, text: str) -> IntermediateLLMResponse | NoReturn:
        """Parse out desired JSON object from LLM response."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find('{')
            end = text.find('}')
            if start != -1 and end != -1 and start < end:
                try:
                    data = json.loads(text[start:end+1])
                except json.JSONDecodeError:
                    raise ValueError("Could not parse JSON from response.")
            else:
                raise ValueError("Could not parse JSON from response.")
            
        if isinstance(data, dict) and "intermediate_answer" in data and "next_query" in data:
            return IntermediateLLMResponse(intermediate_answer=data["intermediate_answer"], next_query=data["next_query"])
        raise ValueError("Could not parse JSON from response.")
