from llm_behaviours.pre_retrieval.base import QueryRewriter

class IdentityRewriter(QueryRewriter):
    """Unaltered - original query returned."""

    DESCRIPTION = "Don't rewrite the query, the original is optimal."

    def rewrite(self, query: str) -> list[str]:
        return [query]