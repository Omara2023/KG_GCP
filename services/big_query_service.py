from services.mixins import EnvMixin
from clients.biq_query_client import BigQueryClient
from models.log_entry import LogEntry

class BigQueryService(EnvMixin):
    def __init__(self):
        self.project_id = self._get_env("GCP_PROJECT_ID")
        self.dataset_id = self._get_env("GCP_BQ_DATASET_ID")
        self.table_id = self._get_env("GCP_BQ_TABLE_ID")
        self.client = BigQueryClient(self.project_id, self.dataset_id, self.table_id)

    def log_query(self, log_entry: LogEntry):
        self.client.insert_rows(log_entry)
