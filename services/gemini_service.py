import os 
import logging
from gemini_caller import GeminiCaller

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        self.caller = GeminiCaller(
            os.environ.get("GCP_PROJECT_ID"),
            os.environ.get("GCP_ENGINE_ID"),   
            "gemini-1.5-flash"
        )

    def response(self, query: str, contexts: list):
        if not contexts:
            logger.info("No relevant contexts found. Responding without grounding.")
            return self.caller.generate_response(query, [])
        return self.caller.generate_response(query, contexts)