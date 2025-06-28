import os
from vertex_ai_caller import VertexAICaller

class VertexService:
    def __init__(self):
        self.caller = VertexAICaller(
            os.environ.get("GCP_PROJECT_ID"),
            os.environ.get("GCP_VERTEX_LOCATION"),
            os.environ.get("GCP_ENGINE_ID") 
        )

    def search(self, query: str):
        return self.caller.run_vertex_ai_search(query)