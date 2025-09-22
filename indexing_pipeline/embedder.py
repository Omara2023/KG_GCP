from langchain_huggingface import HuggingFaceEmbeddings

class Embedder:
    """Class to turn chunks into dense vectors via embedding model."""

    def __init__(self, model_name="all-mpnet-base-v2"): #TODO - change this to use dependency injections, preparing a set embedder to pass in and env based model name.
        self.embedder = HuggingFaceEmbeddings(model_name=model_name)

    def embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        return self.embedder.embed_documents(chunks)
    
    def embed_query(self, query: str) -> list[float]:
        return self.embed_query(query)
