import logging
from services.mixins import EnvMixin
from clients.gemini_client import GeminiClient
from models.context import RetrievedContext
from llm_behaviours.grounded_generator import GroundedAnswerGenerator
from llm_behaviours.step_back_rewriter import StepBackRewriter

class GeminiService(EnvMixin):
    def __init__(self):
        self.project_id = self._get_env("GCP_PROJECT_ID")
        self.location = self._get_env("GCP_GEMINI_REGION")
        self.client = GeminiClient(project_id=self.project_id, location=self.location, model_name="gemini-1.5-flash")
        self.grounded_generator =  GroundedAnswerGenerator(self.client)
        self.step_back_rewriter = StepBackRewriter(self.client)
        
        self.logger = logging.getLogger(__name__)

    def respond(self, query: str, contexts: list[RetrievedContext]) -> str:
        return self.grounded_generator.generate(query, contexts)
    
    def rewrite_query(self, query: str) -> str:
        return self.step_back_rewriter.rewrite(query)
    
def get_gemini_service() -> GeminiService:
    return GeminiService()