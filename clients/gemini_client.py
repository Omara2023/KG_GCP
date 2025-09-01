import logging
from google import genai
from google.genai import types

class GeminiClient:
    """Wrapper class to generate responses using Gemini, grounded by the retrieved contexts."""

    def __init__(self, project_id, location, system_instruction = None, model_name="gemini-1.5-flash"):
        self.client = genai.Client(vertexai=True, project=project_id, location=location)
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.system_instruction = system_instruction

    def prompt(self, prompt: str) -> str:
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
        
        return generated_text

    def _generate_content_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=4000,
            system_instruction=self.system_instruction
        )
   