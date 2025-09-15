import logging
import uuid
from langchain.schema import Document
from clients.neo4j_client import Neo4jClient
from indexing_pipeline.gcs_pdf_loader import GCSPDFLoader
from indexing_pipeline.embedder import Embedder
from indexing_pipeline.chunker import Chunker

class GraphDBInjector:

    def __init__(self, pdf_loader: GCSPDFLoader, chunker: Chunker, embedder: Embedder, graph_client: Neo4jClient) -> None:
        self.pdf_loader = pdf_loader 
        self.chunker = chunker
        self.embedder = embedder
        self.graph_client = graph_client
        self.logger = logging.getLogger(__name__)

    def index_pdf(self, path: str) -> None:
        """Index one or more PDFs from GCS bucket."""
        documents = self.pdf_loader.load(path)
        if not documents:
            self.logger.warning(f"No documents returned for {path}")
            return

        pdf_id = self._generate_pdf_id()
        
        for doc in documents:
            filename = self._extract_filename(doc)
        
            chunks = self.chunker.chunk_text(doc.page_content, 200, 50)
            docs = [
                Document(page_content=c, metadata={"source": filename})
                for c in chunks
            ]
            embeddings = self.embedder.embed_chunks(docs).tolist()
            to_injector = [
                {"text": c, "embedding": e} for c, e in zip(chunks, embeddings)
            ]

            self.graph_client.store_chunks(pdf_id, filename, to_injector)

    def _extract_filename(self, doc: Document) -> str:
        output = doc.metadata.get("source")
        if output is not None:
            return output
        raise ValueError("Invalid docmument metadata. Failed to extract filename.")

    def _generate_pdf_id(self) -> str:
        return str(uuid.uuid4())