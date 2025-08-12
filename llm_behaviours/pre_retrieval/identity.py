from llm_behaviours.pre_retrieval.base import QueryRewriter

class IdentityRewriter(QueryRewriter):
    """Unaltered - original query returned."""

    def rewrite(self, query: str) -> list[str]:
        return [query]