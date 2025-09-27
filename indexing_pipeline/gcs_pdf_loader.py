import pymupdf
from google.cloud import storage
from langchain.schema import Document

class GCSPDFLoader:
    """Loader to fetch PDFs from a GCS bucket and return LangChain Documents."""

    def __init__(self, bucket_name: str):
        """Expecting path for access to GCS bucket containing PDFs."""
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def load(self, filename: str) -> Document:
        """Fetch a single PDF from GCS and extract contents into LangChain Document."""
        try:
            blob = self.bucket.blob(filename)
            pdf_bytes = blob.download_as_bytes()
            
            with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
                #Pylance might flag this, but it is correct.
                text = "\n".join(page.get_text("text") for page in doc) # type: ignore
                
            return Document(
                page_content=text,
                metadata={"source": f"gs://{self.bucket.name}/{filename}"}
            )
            
        except Exception as e:
            print(f"[Error] Failed to load {filename}: {e}")
            exit(1)

    def load_all(self, prefix: str = "") -> list[Document]:
        """Fetch all PDFs under a GCS prefix and return as Documents."""
        documents = []
        for blob in self.bucket.list_blobs(prefix=prefix):
            if blob.name.lower().endswith(".pdf"):
                documents.append(self.load(blob.name))
        return documents
