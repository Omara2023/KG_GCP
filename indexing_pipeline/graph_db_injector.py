import logging
from langchain.schema import Document
from clients.neo4j_client import Neo4jClient
from indexing_pipeline.pdf_parser import PDFLoader
from indexing_pipeline.embedder import Embedder
from indexing_pipeline.chunker import Chunker

class GraphDBInjector:

    def __init__(self, parser: PDFLoader, chunker: Chunker, embedder: Embedder, graph_client: Neo4jClient) -> None:
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.graph_client = graph_client
        self.logger = logging.getLogger(__name__)


    def index_pdf(self, path: str) -> None:
        text = self.parser.extract_text_from_pdf(path)
        filename = self._extract_filename(path)
        pdf_id = self._generate_pdf_id(filename)
        chunks = self.chunker.chunk_text(text, 200, 50)
        docs = [Document(c, source=filename) for c in chunks]
        embeddings = self.embedder.embed_chunks(docs).tolist()
        to_injector = []
        for c, e in zip(chunks, embeddings):
            to_injector.append({"text": c, "embedding": e})
        self.graph_client.store_chunks(pdf_id, filename, to_injector)

    def _extract_filename(self, file: str) -> str:
        return "hello world"

    def _generate_pdf_id(self, filename: str) -> str:
        return "fum"