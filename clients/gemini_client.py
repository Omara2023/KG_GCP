import logging
from google import genai
from google.genai import types
from models.llm_response import LLMResponse

class GeminiClient:
    """Wrapper class to generate responses using Gemini, grounded by the retrieved contexts."""

    def __init__(self, project_id, location, model_name):
        self.client = genai.Client(vertexai=True, project=project_id, location=location)
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

    def prompt(self, prompt: str) -> LLMResponse:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self._generate_content_config()
            )

            if response.text is None:
                raise ValueError("Gemini didn't return any text.")
            generated_text = response.text
        except Exception as e:
            self.logger.exception(f"Error during Gemini generation: {e}")
            generated_text = "I apologize, but I encountered an error while generating a response."
        
        return LLMResponse(text=generated_text)

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
   