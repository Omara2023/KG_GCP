import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

class Embedder:
    """Class to turn chunks into dense vectors via embedding model."""

    def __init__(self, model_name="all-mpnet-base-v2"):
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    def embed_chunks(self, chunks: list[Document]) -> np.ndarray:
        faiss = FAISS.from_documents(chunks, self.embedding_model)
        return self._faiss_to_array(faiss)
    
    def _faiss_to_array(self, faiss: FAISS) -> np.ndarray:
        index = faiss.index
        ntotal = index.ntotal
        vectors = [index.reconstruct(i) for i in range(ntotal)]
        return np.array(vectors)