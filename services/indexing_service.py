from clients.neo4j_client import Neo4jClient
from indexing_pipeline.gcs_pdf_loader import GCSPDFLoader
from indexing_pipeline.chunker import Chunker
from indexing_pipeline.embedder import Embedder
from indexing_pipeline.graph_db_injector import GraphDBInjector

async def handle_event(data: dict, client: Neo4jClient): #Consider making the graph retriever interface vendor agnostic.
    base = "google.cloud.storage.object.v1"
    type: str = data["type"]

    if not type.startswith(base):
        raise ValueError(f"Cannot handle this type of event: {type}")
    
    cloud_storage_type = type.split(".")[-1]

    match cloud_storage_type:
        case "finalized":
            await finalised(data, client)
        case "deleted":
            await deleted(data, client)
        case _:
            raise ValueError(f"Invalid Cloud Storage event: {type}")
    
async def finalised(data: dict, client: Neo4jClient):
    """Fires whenever a new object is created in bucket."""
    subject = data["subject"]
    


async def deleted(data: dict, client: Neo4jClient):
    """Called when a file is deleted. Note, GCS filename change is a compound, create then delete old."""
    pass
