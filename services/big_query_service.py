from clients.biq_query_client import BigQueryClient
from models.log_entry import LogEntry

class BigQueryService:
    def __init__(self, client: BigQueryClient):
        self.client = client

    def log_query(self, log_entry: LogEntry):
        self.client.insert_rows(log_entry)

