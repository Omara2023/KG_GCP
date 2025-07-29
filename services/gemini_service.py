import logging
from services.mixins import EnvMixin
from clients.gemini_client import GeminiClient
from models.context import RetrievedContext

class GeminiService(EnvMixin):
    def __init__(self):
        self.project_id = self._get_env("GCP_PROJECT_ID")
        self.location = self._get_env("GCP_GEMINI_REGION")
        self.client = GeminiClient(project_id=self.project_id, location=self.location, model_name="gemini-1.5-flash")
        self.logger = logging.getLogger(__name__)

    def respond(self, query: str, contexts: list[RetrievedContext]) -> dict:
        if not contexts:
            self.logger.info("No relevant contexts found. Responding without grounding.")
            return self.client.generate_response(query, [])
        return self.client.generate_response(query, contexts)