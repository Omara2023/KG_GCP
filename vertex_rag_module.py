import logging
import json
from typing import List, Dict
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine

logger = logging.getLogger(__name__)
            
class VertexAICaller:
    """Wrapper class that calls Vertex AI Search app to retrieve relevant context."""

    def __init__(self, project_id: str, location: str, engine_id: str):
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
            response = client.search(request=request)
            if (output := self._extract_and_format_summary(response)):
                retrieved_contexts.append(output)
            
            if (output := self._extract_snippets_and_format(response)):
                retrieved_contexts.extend(output)
                
        except Exception as e:
            logger.exception(f"Error during Vertex AI Search retrieval: {e}")
            exit(1)
        
        finally:
            logger.info(f"Final retrieved contexts count: {len(retrieved_contexts)}")
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
    
    def _extract_and_format_summary(self, response) -> Dict[str, str]:
        """Read summary from SearchPager and return cleaned summary."""
        first_response = response._response
        if first_response.summary and first_response.summary.summary_text:
            logger.info("Added overall summary to contexts.")
            return {"type": "summary", "content": first_response.summary.summary_text}
        else:
            logger.info("No overall summary found in response.")
        
    def _extract_snippets_and_format(self, response) -> List[Dict[str, str]]:
        """Read snippets from Search Pager and return cleaned entries."""
        output = list()
        snippets_added = False
        for result in response:
            logger.info(response)
            continue
            doc = result.document
            if doc and doc.json_data:
                try:
                    data = json.loads(doc.json_data)
                    text = data.get("content")
                    if text:
                        output.append({"type": "snippet", "content": text})
                        snippets_added = True
                except json.JSONDecodeError:
                    logger.warning("Malformed json_data in document.")
        logger.info("Added snippets to contexts." if snippets_added else "No snippets added.")
        return output

