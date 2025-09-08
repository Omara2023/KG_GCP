import logging
from llm_behaviours.pre_retrieval.base import QueryRewriter
from mixins.llm_json_parser import LLMJSONParserMixin
from clients.gemini_client import GeminiClient

class QueryRewriteRouter(LLMJSONParserMixin):
    """On being fed input query decides how best to optimise the query from the given strategies."""

    def __init__(self, rewriters: dict[str, QueryRewriter], llm_client: GeminiClient):
        self.rewriters = rewriters
        self.llm_client = llm_client
        self.llm_client.system_instruction = "You are an expert in information sciences. You will be deciding on how best to rewrite/optimise a given query for RAG retrieval from a corpus of documents."
        self.logger = logging.getLogger(__name__)

    def select_rewriter(self, query: str) -> QueryRewriter:
        """LLM decides which rewriter would be best for input query."""
        prompt = self._construct_prompt(query)
        output = self.llm_client.prompt(prompt)
        judegment_dict = self._safe_parse_json(output) 
        key = judegment_dict["verdict"]
        if key not in self.rewriters.keys(): 
            raise ValueError("Incorrect QueryRewriteRouter LLM Output.")
        self._log_outcome(judegment_dict)
        return self.rewriters[key]

    def _construct_prompt(self, query: str) -> str:
        rewriters_text = "\n".join(f"{k}: {v}." for k, v in self.rewriters.items())
        return f"""
            Decide which of the following query rewriters is most appropriate to optimise this query for RAG retrieval.

            Query: {query}

            Proposed Rewriters:
            {rewriters_text}

            Return the key of the most appropriate option with JSON key:
            {{"verdict": "<KEY>"}}

            where <KEY> is the option which is best suited for retrieval in an attempt to answer the query.

            To reiterate, your answer should be a formatted JSON object that Python's json.loads() can process directly.
        """

    def _log_outcome(self, judgement_dict: dict) -> None:    
        self.logger.info(f"Proposed rewrite strategy: {judgement_dict['verdict']}.")