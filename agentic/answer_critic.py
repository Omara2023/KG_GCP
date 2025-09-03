import logging
from mixins.llm_json_parser import LLMJSONParserMixin
from clients.gemini_client import GeminiClient

class AnswerCritic(LLMJSONParserMixin):
    """Evaluates proposed answer to query and coordinates next action."""

    def __init__(self, llm_client: GeminiClient):
        self.llm_client = llm_client
        self.llm_client.system_instruction = "You are an expert in world knowledge. You will be assessing whether a response answers the given question in a satisfactory manner."
        self.logger = logging.getLogger(__name__)

    def judge(self, query: str, answer: str) -> dict:
        """LLM decides whether or not the query has been sufficiently answered."""
        prompt = self._construct_prompt(query, answer)
        output = self.llm_client.prompt(prompt)
        judegment_dict = self._safe_parse_json(output) 
        if judegment_dict["verdict"] not in ["satisfactory", "unsatisfactory"]: #replace with enumerated type VERDICT
            raise ValueError("Incorrect AnswerCritic LLM Output.")
        self._log_outcome(query, answer, judegment_dict)
        return judegment_dict

    def _construct_prompt(self, query: str, answer: str) -> str:
        return (
            "Decide whether this answer properly answers this query or not.\n"
            f"Query: {query}\n"
            f"Proposed Answer: {answer}\n "
            "If the answer is satisfactory you will reply back with JSON 'verdict': 'satisfactory'."
            "Should the answer not be satisfactory you will reply with JSON object 'verdict: 'unsatisfactory' and 'follow_up': '<QUERY>',"
            "where <QUERY> is a follow up question of your choosing that is best suited for another round of retrieval in an attempt to answer the original query."
            "To reiterate, your answer should be a formatted JSON object that py's json.loads() can process directly."
        )
    
    def _log_outcome(self, query: str, answer: str, judgement_dict: dict) -> None:
        if judgement_dict["verdict"] == "satisfactory":
            self.logger.info(f"Answer: {answer} to query: {query}, is acceptable.")
        else:
            self.logger.info(f"Answer: {answer} to query: {query}, is NOT acceptable.")
            self.logger.info(f"Follow up query: {judgement_dict['follow_up']}")