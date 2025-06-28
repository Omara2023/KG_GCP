import logging
from typing import List, Dict, Any
from google import genai
from google.genai import types

class GeminiCaller:
    """Wrapper class to generate responses using Gemini, grounded by the retrieved contexts."""

    def __init__(self, project_id, location, model_name, logger: logging.Logger = None):
        self.client = genai.Client(vertexai=True, project=project_id, location=location)
        self.model_name = model_name
        self.logger = logger or logging.getLogger(__name__)

    def generate_response(self, user_query: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Returns a dictionary with the generated answer and a list of all unique citations."""
        full_prompt = self._construct_prompt(user_query, contexts)
        self.logger.info(f"Gemini prompt length: {len(full_prompt)}...")

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

        self.logger.info(f"Generated output length: {len(generated_text)}.")
        return {"response": generated_text}

    def _construct_prompt(self, user_query: str, contexts: List[Dict[str, Any]]) -> str:
        """String together user query and context into a LLM-ready prompt."""
        prompt_parts = [
            f"Answer the user's question, grounding your answer in the provided context. Do not make up information. If the provided context lacks the answer, state so before answering.",
            "\n\n--- Retrieved Contexts ---"
        ]

        for i, context in enumerate(contexts):
            prompt_parts.append(f"\nContext {i+1}:\n{context['content']}")
        prompt_parts.append("\n-------------------------\n")
        prompt_parts.append(f"User's Question: {user_query}")

        full_prompt = "\n".join(prompt_parts)
        return full_prompt

    def _generate_content_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig({"temperature": 0.4, "max_output_tokens": 4000})


   