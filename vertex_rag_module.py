import os
import vertexai
from typing import List, Dict
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id") #project_id: Your Google Cloud project ID.
LOCATION = os.environ.get("GCP_LOCATION", "global") #location: The region of your search engine.
ENGINE_ID = os.environ.get("GCP_ENGINE_ID") #engine_id: The ID of your Vertex AI Search app (engine).         
            
class VertexAICaller:
    """Wrapper class that calls Vertex AI Search app to retrieve relevant context."""

    def __init__(self, project_id: str = PROJECT_ID, location: str = LOCATION, engine_id: str = ENGINE_ID):
        self.project_id = project_id
        self.location = location
        self.engine_id = engine_id
        try:
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            print(f"Vertex AI initialized for project {PROJECT_ID} in location {LOCATION}")
        except Exception as e:
            print(f"Error initializing Vertex AI: {e}")
            # Handle this error appropriately in a production environment (e.g., exit, log, health check fail)

    def run_vertex_ai_search(self, query: str) -> List[Dict[str, str]]:
        """Perform similarity search on Vertex AI search app, returing results."""
        print(f"Attempting to retrieve context for query: '{query}'")
        client_options = (ClientOptions(api_endpoint=f"{self.location}-discoveryengine.googleapis.com") if self.location != "global" else None)

        client = discoveryengine.SearchServiceClient(client_options=client_options)

        serving_config = (
            f"projects/{self.project_id}/locations/{self.location}/collections/default_collection/"
            f"engines/{self.engine_id}/servingConfigs/default_config"
        )

        content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
            snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                return_snippet=True  # Request snippets for direct context
            ),
            summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
                summary_result_count=3,  # How many summaries to return
                include_citations=True,  # Crucial for grounding and transparency
                ignore_adversarial_query=True,
                ignore_non_summary_seeking_query=True,
                model_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelSpec(
                    version="stable", # Use a stable LLM version for summarization
                ),
            ),
        )

        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=query,
            page_size=5,  # Number of raw search results to consider
            content_search_spec=content_search_spec,
            query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
                condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
            ),
            spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
                mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
            ),
        )

        retrieved_contexts = []
        try:
            # page_result here is the TOP-LEVEL SearchResponse object
            response = client.search(request=request)

            # 1. Process the overall summary first (if it exists)
            if response.summary and response.summary.summary_text:
                summary_text = response.summary.summary_text
                citations = []
                if response.summary.citation_metadata:
                    for citation_source in response.summary.citation_metadata.citation_sources:
                        if citation_source.uri:
                            citations.append(citation_source.uri)
                retrieved_contexts.append({
                    "content": summary_text,
                    "citations": citations,
                    "type": "summary" # Add type to distinguish summary from snippets
                })
                print(f"DEBUG: Added overall summary to contexts.")
            else:
                print("DEBUG: No overall summary found in response.")


            # 2. Iterate through individual search results for snippets
            for search_result_item in response.results: # <-- Iterate over .results
                # Each search_result_item is a discoveryengine.SearchResponse.SearchResult
                if search_result_item.snippet and search_result_item.snippet.snippet:
                    snippet_text = search_result_item.snippet.snippet
                    source_url = search_result_item.document.uri if search_result_item.document and search_result_item.document.uri else "N/A"
                    retrieved_contexts.append({
                        "content": snippet_text,
                        "citations": [source_url] if source_url != "N/A" else [],
                        "type": "snippet" # Add type to distinguish snippets from summary
                    })
                    print(f"DEBUG: Added snippet from {source_url} to contexts.")
                else:
                    print("DEBUG: No snippet found for a search result item.")


        except Exception as e:
            print(f"Error during Vertex AI Search retrieval: {e}")
            # In a real app, you might want to return an error or empty context
        
        print(f"DEBUG: Final retrieved contexts count: {len(retrieved_contexts)}")
        return retrieved_contexts