from json import loads
from enums.retriever_types import RetrieverTypes
from clients.gemini_client import GeminiClient

class RetrievalRouter:
    """Decide which retrieval strategy should be used for a query."""
    def __init__(self, llm_client: GeminiClient):
        self.llm = llm_client
        self.llm.system_instruction = "You are an expert decision maker, deciding what type of RAG lookup strategy to use in order to answer user queries factually and usefully."

    def get_retriever(self, query: str) ->  RetrieverTypes:
        prompt = self._contruct_prompt(query)
        output = self.llm.prompt(prompt)
        option = loads(output)["choice"]
        return self._string_to_retriever_type(option)
    
    def _contruct_prompt(self, query: str) -> str:
        return (
            "Decide which retrieval strategy is best suited for a RAG based system to answer the user query.\n"
            "Types:\n"
            f"{', '.join(f'RetrieverTypes.{i.value}' for i in RetrieverTypes)}\n"
            f"User query: {query}\n"
            "Return your answer as JSON with key 'choice' and value RetrieverTypes.<TYPE>"
        )

    def _string_to_retriever_type(self, input: str) -> RetrieverTypes:
        match input:
            case RetrieverTypes.SIMPLE.value:
                return RetrieverTypes.SIMPLE
            case _:
                raise ValueError("Invalid retriever type returned from LLM.")