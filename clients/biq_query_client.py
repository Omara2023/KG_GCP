import json
import logging
from google.cloud import bigquery
from models.log_entry import LogEntry

class BigQueryClient:
    """Class to insert BigQuery logging records."""
    def __init__(self, project_id: str, dataset_id: str, table_id: str):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.client = bigquery.Client()
        self.logger = logging.getLogger(__name__)

    def insert_rows(self, row: LogEntry) -> None:
        data = row.to_dict()
        self.logger.info(json.dumps(data, indent=2))
        table = self._table()
        errors = self.client.insert_rows_json(table=table, json_rows=[data])

        if errors:
            self.logger.error("Encountered errors while inserting row: ", errors)
        else:
            self.logger.info("Inserted entry successfully!")

    def _table(self) -> str:
        """Return fully qualified BigQuery table_id."""
        return f"{self.project_id}.{self.dataset_id}.{self.table_id}"