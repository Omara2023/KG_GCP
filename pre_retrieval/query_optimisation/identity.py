from pre_retrieval.query_optimisation.base import QueryRewriter

class IndentityStrategy(QueryRewriter):
    """Unaltered - original query returned."""

    def rewrite(self, query: str) -> list[str]:
        return [query]