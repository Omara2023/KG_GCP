import logging
from typing import List, Dict, Any
from vertexai.generative_models import GenerativeModel

logger = logging.getLogger(__name__)

class GeminiCaller:
    """Wrapper class to generate responses using Gemini 1.5 Flash, grounded by the retrieved contexts."""

    def __init__(self, model_name):
        self.model_name = model_name

    def construct_prompt(self, user_query: str, contexts: List[Dict[str, Any]]) -> str:
        """String together user query and context into a LLM-ready prompt."""
        prompt_parts = [
            f"Based on the following information, answer the user's question. If the information does not contain the answer, state that you cannot answer based on the provided context. Do not make up information.",
            "\n\n--- Retrieved Contexts ---"
        ]

        for i, context in enumerate(contexts):
            prompt_parts.append(f"\nContext {i+1}:\n{context['content']}")
        prompt_parts.append("\n-------------------------\n")
        prompt_parts.append(f"User's Question: {user_query}")

        full_prompt = "\n".join(prompt_parts)
        return full_prompt

    def generate_response(self, user_query: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Returns a dictionary with the generated answer and a list of all unique citations."""
        model = GenerativeModel(self.model_name)

        full_prompt = self.construct_prompt(user_query, contexts)
        logger.info(f"Full prompt sent to Gemini is of length: {len(full_prompt)}...")
        logger.info(f"Actual prompt: {len(full_prompt)}...")

        CHUNK_SIZE = 150
        for i in range(0, len(full_prompt), CHUNK_SIZE):
            logger.info(f"{full_prompt[i: i+CHUNK_SIZE]}...")

        try:
            response = model.generate_content(
                full_prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 1024}
            )

            generated_text = response.text
        except Exception as e:
            logger.exception(f"Error during Gemini generation: {e}")
            generated_text = "I apologize, but I encountered an error while generating a response."

        logger.info(f"Length of generated output: {len(generated_text)}.")
        return {"response": generated_text,}