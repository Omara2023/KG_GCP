import os
import vertexai
import logging
from typing import List, Dict, Optional, Union
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
from google.cloud.discoveryengine_v1.types import SearchResponse

logger = logging.getLogger(__name__)

PROJECT_ID = os.environ.get("GCP_PROJECT_ID") # Your Google Cloud project ID.
LOCATION = os.environ.get("GCP_LOCATION") # The region of your search engine.
ENGINE_ID = os.environ.get("GCP_ENGINE_ID") # The ID of your Vertex AI Search app (engine).         
            

class VertexAICaller:
    def __init__(self, project_id: str = PROJECT_ID, location: str = LOCATION, engine_id: str = ENGINE_ID):
        self.project_id = project_id
        self.location = location
        self.engine_id = engine_id
        try:
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            logger.info(f"Vertex AI initialized for project {PROJECT_ID} in location {LOCATION}")
        except Exception as e:
            logger.error(f"Error initializing Vertex AI: {e}")

    def run_vertex_ai_search(self, query: str) -> List[Dict[str, Union[str, List[str]]]]:
        """Query Vertex AI Search and return summarized + snippet content."""
        logger.info(f"Retrieving context for query: '{query}'")

        client = discoveryengine.SearchServiceClient(
            client_options=ClientOptions(
                api_endpoint=f"{self.location}-discoveryengine.googleapis.com"
            ) if self.location != "global" else None
        )

        request = self._build_request(query)
        response = client.search(request=request)

        try:
            first_page = next(response.pages)
        except Exception as e:
            logger.exception("Failed to access first page of results.")
            return []

        retrieved_contexts = []

        # 1. Process summary if present
        summary_context = self._extract_summary(first_page)
        if summary_context:
            retrieved_contexts.append(summary_context)

        # 2. Process search results (snippets)
        snippet_contexts = self._extract_snippets(response)
        retrieved_contexts.extend(snippet_contexts)

        logger.debug(f"Total contexts retrieved: {len(retrieved_contexts)}")
        return retrieved_contexts

    def _build_request(self, query: str) -> discoveryengine.SearchRequest:
        """Builds the SearchRequest for Vertex AI Search."""
        return discoveryengine.SearchRequest(
            serving_config=(
                f"projects/{self.project_id}/locations/{self.location}/collections/default_collection/"
                f"engines/{self.engine_id}/servingConfigs/default_config"
            ),
            query=query,
            page_size=5,
            content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
                snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                    return_snippet=True
                ),
                summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
                    summary_result_count=3,
                    include_citations=True,
                    ignore_adversarial_query=True,
                    ignore_non_summary_seeking_query=True,
                    model_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelSpec(
                        version="stable"
                    ),
                ),
            ),
            query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
                condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO
            ),
            spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
                mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
            ),
        )

    def _extract_summary(self, response: SearchResponse) -> Optional[Dict[str, Union[str, List[str]]]]:
        """Extracts summary and its citations from the first page response."""
        summary = response.summary
        if not summary or not summary.summary_text:
            logger.debug("No summary found in response.")
            return None

        citations = []
        citation_meta = getattr(summary, "CitationMetadata", None)
        if citation_meta:
            for source in citation_meta.citation_sources:
                if source.uri:
                    citations.append(source.uri)

        logger.debug("Extracted summary from response.")
        return {
            "content": summary.summary_text,
            "citations": citations,
            "type": "summary"
        }

    def _extract_snippets(self, pager: discoveryengine.SearchPager) -> List[Dict[str, Union[str, List[str]]]]:
        """Extracts snippets from each SearchResult in the response pager."""
        snippets = []

        for result in pager:
            snippet_text = getattr(result.snippet, "snippet", None)
            if not snippet_text:
                logger.debug("Search result has no snippet.")
                continue

            uri = getattr(result.document, "uri", "N/A") if result.document else "N/A"
            snippets.append({
                "content": snippet_text,
                "citations": [uri] if uri != "N/A" else [],
                "type": "snippet"
            })
            logger.debug(f"Added snippet from {uri}")

        return snippets
