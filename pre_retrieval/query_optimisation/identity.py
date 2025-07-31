from models.user_query import UserQuery
from pre_retrieval.query_optimisation.base import QueryRewriter

class Indentity(QueryRewriter):
    """Unaltered - original query returned."""

    def rewrite_query(self, query: UserQuery) -> list[UserQuery]:
        return [query]