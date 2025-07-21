import os 
import logging
from clients.gemini_client import GeminiClient

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        self.project_id = self._get_env("GCP_PROJECT_ID")
        self.location = self._get_env("GCP_GEMINI_REGION")
        self.client = GeminiClient(project_id=self.project_id, location=self.location, model_name="gemini-1.5-flash")

    def _get_env(self, var_name: str) -> str:
        value = os.environ.get(var_name)
        if not value:
            raise ValueError(f"Missing required environment variable: {var_name}")
        return value

    def respond(self, query: str, contexts: list):
        if not contexts:
            logger.info("No relevant contexts found. Responding without grounding.")
            return self.client.generate_response(query, [])
        return self.client.generate_response(query, contexts)