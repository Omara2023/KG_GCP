from pre_retrieval.query_optimisation.base import QueryRewriter

class Indentity(QueryRewriter):
    """Unaltered - original query returned."""

    def rewrite(self, query: str) -> list[str]:
        return [query]