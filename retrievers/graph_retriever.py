import logging
from retrievers.base import Retriever

class GraphRetriever(Retriever):
    """Application-level wrapper that adapts GraphDB search results."""

    