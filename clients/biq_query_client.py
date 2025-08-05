import logging
from google.cloud import bigquery
from models.log_entry import LogEntry

class BigQueryClient:
    """Class to insert BigQuery logging records."""
    def __init__(self, project_id: str, dataset_id: str, table_id: str):
        self.table = f"{project_id}.{dataset_id}.{table_id}"
        self.client = bigquery.Client()
        self.logger = logging.getLogger(__name__)

    def insert_rows(self, row: LogEntry) -> None:
        data = row.model_dump()
        errors = self.client.insert_rows_json(table=self.table, json_rows=[data])

        if errors:
            self.logger.error("Encountered errors while inserting row: ", errors)
        else:
            self.logger.info("Inserted entry successfully!")

