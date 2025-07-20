import logging
from typing import List, Dict
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine

class VertexAIRagCaller:
    """Wrapper class that calls Vertex AI RAG engine to retrieve context."""

    def __init__(self, project_id: str, location: str, rag_corpus_id: str):
        self.project_id = project_id
        self.location = location
        self.rag_corpus_id = rag_corpus_id
        self.logger = logging.getLogger(__name__)

    def run_vertex_ai_search(self, query: str) -> List[Dict[str, str]]:
        """Perform similarity search on Vertex AI search app, returing results."""        
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
            self.logger.exception(f"Error during Vertex AI Search retrieval: {e}")
            exit(1)
        
        return retrieved_contexts
    
    def _rag_corpus_resoruce(self) -> str:
        """Return fully qualified name of Rag corpus."""
        return f"projects/{self.project_id}/locations/{self.location}/locations/ragCorpora/{self.rag_corpus_id}"

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
            self.logger.info("Added overall summary to contexts.")
            return {"type": "summary", "content": first_response.summary.summary_text}
        else:
            self.logger.info("No overall summary found in response.")
        
    def _extract_snippets_and_format(self, response) -> List[Dict[str, str]]:
        """Read snippets from Search Pager and return cleaned entries."""
        output = []

        for result in response:
            doc = result.document
            if not doc:
                continue

            derived_fields = doc.derived_struct_data
            derived = dict(derived_fields)

            if "snippets" in derived:
                snippets = derived.get("snippets", [])
                for item in snippets:
                    item_dict = dict(item.items())
                    if "snippet" in item_dict:
                        t = item_dict["snippet"]
                        if t:
                            output.append({"type": "snippet", "content": t})
        
        self.logger.info(f"Added {(num := len(output))} snippet{"s" if num > 1 else ""} to contexts.")
        return output


