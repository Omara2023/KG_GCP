from clients.gemini_client import GeminiClient
from models.context import RetrievedContext

class GroundedAnswerGenerator:
    """Class responsible for the main augmented generation step."""  
    
    def __init__(self, llm_client: GeminiClient):
        self.llm = llm_client

    def generate(self, query: str, contexts: list[RetrievedContext]) -> str:
        if not contexts:
            return self._ask_without_context(query)
        return self._ask_with_context(query, contexts)
        
    def _ask_with_context(self, user_query: str, contexts: list[RetrievedContext]) -> str:
        prompt_parts = ["\n\n--- Retrieved Contexts ---"]
        for i, context in enumerate(contexts):
            prompt_parts.append(f"\nContext {i+1}:\nSource:{context.source_file}\nText:{context.text}")
        prompt_parts.append("\n-------------------------\n")
        prompt_parts.append("Using the above contexts, answer the following question as accurately as possible.")
        prompt_parts.append(f"\n\nUser's Question: {user_query}")
        prompt = "\n".join(prompt_parts)
        return self.llm.prompt(prompt)

    def _ask_without_context(self, user_query: str) -> str:
        prompt = (
            "No external context is available for this query. Please answer using your own knowledge, "
            "and clearly state that context was not provided.\n\n"
            f"User's Question: {user_query}"
        )
        return self.llm.prompt(prompt)