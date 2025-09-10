import os
import pymupdf
from langchain.schema import Document

class PDFLoader:
    """Class to read text from PDFs in a directory."""

    def __init__(self, source_uri: str):
        """Expecting path for access to GCS bucket containing PDFs."""
        self.source_uri = source_uri

    def extract_text_from_pdf(self, filename: str) -> str:
        """Extracts all text from a single PDF file using PyMuPDF."""
        try:
            full_path = os.path.join(self.source_uri, filename)
            doc = pymupdf.open(full_path)
            text = "\n".join(page.get_text("text") for page in doc)
            return text
        except Exception as e:
            print(f"[Error] Failed to load {full_path}: {e}")
            exit(1)