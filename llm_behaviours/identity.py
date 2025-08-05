from llm_behaviours.base import QueryRewriter

class IdentityRewriter(QueryRewriter):
    """Unaltered - original query returned."""

    def rewrite(self, query: str) -> list[str]:
        return [query]