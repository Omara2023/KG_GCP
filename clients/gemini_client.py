import logging
from typing import List, Dict, Any
from google import genai
from google.genai import types
from models.context import RetrievedContext

class GeminiClient:
    """Wrapper class to generate responses using Gemini, grounded by the retrieved contexts."""

    def __init__(self, project_id, location, model_name):
        self.client = genai.Client(vertexai=True, project=project_id, location=location)
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

    def generate_response(self, user_query: str, contexts: List[RetrievedContext]) -> Dict[str, Any]:
        """Returns a dictionary with the generated answer and a list of all unique citations."""
        full_prompt = self._construct_prompt(user_query, contexts)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=self._generate_content_config()
            )
            generated_text = response.text
        except Exception as e:
            self.logger.exception(f"Error during Gemini generation: {e}")
            generated_text = "I apologize, but I encountered an error while generating a response."

        return {"response": generated_text}

    def _construct_prompt(self, user_query: str, contexts: List[RetrievedContext]) -> str:
        """Builds the full prompt for Gemini, adapting based on context availability."""
        if contexts:
            return self._prompt_with_context(user_query, contexts)
        else:
            return self._prompt_without_context(user_query)

    def _prompt_with_context(self, user_query: str, contexts: List[RetrievedContext]) -> str:
        prompt_parts = ["\n\n--- Retrieved Contexts ---"]
        for i, context in enumerate(contexts):
            prompt_parts.append(f"\nContext {i+1}:\nSource:{context.source_file}\nText:{context.text}")
        prompt_parts.append("\n-------------------------\n")
        prompt_parts.append("Using the above contexts, answer the following question as accurately as possible.")
        prompt_parts.append(f"\n\nUser's Question: {user_query}")
        return "\n".join(prompt_parts)

    def _prompt_without_context(self, user_query: str) -> str:
        return (
            "No external context is available for this query. Please answer using your own knowledge, "
            "and clearly state that context was not provided.\n\n"
            f"User's Question: {user_query}"
        )

    def _generate_content_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=4000,
            system_instruction=[
                "You are an expert assistant. Use the retrieved contexts provided to answer the user's question as faithfully as possible. "
                "Do not invent facts or speculate. If the contexts contain relevant information, use them. "
                "If the contexts do not include enough information to answer the question fully, you may rely on your own general knowledge to fill in the gaps, "
                "but clearly indicate when this is the case. If the question cannot be answered with the context or your own knowledge, say so."
            ]
        )
   