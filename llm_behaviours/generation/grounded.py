from clients.gemini_client import GeminiClient
from models.context import RetrievedContext

class GroundedAnswerGenerator:
    """Class responsible for the main augmented generation step."""  
    
    def __init__(self, llm_client: GeminiClient):
        self.llm = llm_client

    def generate(self, query: str, contexts: list[RetrievedContext]) -> str:
        if not contexts:
            raise ValueError("No contexts given to grounded answer generator.")
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
