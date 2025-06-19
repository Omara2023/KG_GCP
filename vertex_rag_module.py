import os
import vertexai
import logging
from typing import List, Dict
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine

logger = logging.getLogger(__name__)

PROJECT_ID = os.environ.get("GCP_PROJECT_ID") # Your Google Cloud project ID.
LOCATION = os.environ.get("GCP_LOCATION") # The region of your search engine.
ENGINE_ID = os.environ.get("GCP_ENGINE_ID") # The ID of your Vertex AI Search app (engine).         
            
class VertexAICaller:
    """Wrapper class that calls Vertex AI Search app to retrieve relevant context."""

    def __init__(self, project_id: str = PROJECT_ID, location: str = LOCATION, engine_id: str = ENGINE_ID):
        try:
            vertexai.init(project=PROJECT_ID, location=LOCATION)
        except Exception as e:
            logger.error(f"Error initializing Vertex AI: {e}")
        else:
            logger.info(f"Vertex AI initialized for project {PROJECT_ID} in location {LOCATION}")
        finally:
            self.project_id = project_id
            self.location = location
            self.engine_id = engine_id

    def run_vertex_ai_search(self, query: str) -> List[Dict[str, str]]:
        """Perform similarity search on Vertex AI search app, returing results."""
        logger.info(f"Attempting to retrieve context for query: '{query}'")
        
        client_options = (ClientOptions(api_endpoint=f"{self.location}-discoveryengine.googleapis.com") if self.location != "global" else None)
        client = discoveryengine.SearchServiceClient(client_options=client_options)

        serving_config = self._serving_config()
        content_search_spec = self._prepare_content_search_spec()
        request = self._prepare_search_request(serving_config, query, content_search_spec)

        retrieved_contexts = []
        try:
            # page_result here is the TOP-LEVEL SearchResponse object
            response = client.search(request=request)
            
            # 1. Process the overall summary first (if it exists)
            first_response = response._response  # Access the first page response directly

            if first_response.summary and first_response.summary.summary_text:
                summary_text = first_response.summary.summary_text
                citations = []
                if first_response.summary.citation_metadata:
                    for citation_source in first_response.summary.citation_metadata.citation_sources:
                        if citation_source.uri:
                            citations.append(citation_source.uri)
                retrieved_contexts.append({
                    "content": summary_text,
                    "citations": citations,
                    "type": "summary"  # Add type to distinguish summary from snippets
                })
                logger.debug("DEBUG: Added overall summary to contexts.")
            else:
                logger.debug("DEBUG: No overall summary found in response.")

            # 2. Iterate through all SearchResult items across pages
            for search_result_item in response:
                if search_result_item.snippet and search_result_item.snippet.snippet:
                    snippet_text = search_result_item.snippet.snippet
                    source_url = (
                        search_result_item.document.uri
                        if search_result_item.document and search_result_item.document.uri
                        else "N/A"
                    )
                    retrieved_contexts.append({
                        "content": snippet_text,
                        "citations": [source_url] if source_url != "N/A" else [],
                        "type": "snippet"  # Add type to distinguish snippets from summary
                    })
                    logger.debug(f"DEBUG: Added snippet from {source_url} to contexts.")
                else:
                    logger.debug("DEBUG: No snippet found for a search result item.")


        except Exception as e:
            logger.exception(f"Error during Vertex AI Search retrieval: {e}")
            # In a real app, you might want to return an error or empty context
        
        logger.debug(f"DEBUG: Final retrieved contexts count: {len(retrieved_contexts)}")
        return retrieved_contexts
    
    def _serving_config(self) -> str:
        """Return fully qualified serving config."""
        output = (
            f"projects/{self.project_id}/locations/{self.location}/collections/default_collection/"
            f"engines/{self.engine_id}/servingConfigs/default_config"
        )
        return output

    def _prepare_content_search_spec(self) -> discoveryengine.SearchRequest.ContentSearchSpec:
        """Instantiate and resturn ContentSearchSpec Class."""
        output = discoveryengine.SearchRequest.ContentSearchSpec(
            snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(return_snippet=True),
            summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
                summary_result_count=3,  
                include_citations=True,
                ignore_adversarial_query=True,
                ignore_non_summary_seeking_query=True,
                model_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelSpec(version="stable",),
            ),
        )
        return output
    
    def _prepare_search_request(self, serving_config: str, query: str, content_search_spec: discoveryengine.SearchRequest.ContentSearchSpec) -> discoveryengine.SearchRequest:
        """Factory meothod to create SearchRequest Class."""
        output = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=query,
            page_size=5,
            content_search_spec=content_search_spec,
            query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,),
            spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO),
        )
        return output