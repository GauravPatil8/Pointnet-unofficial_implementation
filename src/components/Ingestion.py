import os
from src.constants import CONFIG_PATH, PROJECT_ROOT
from src.utils.common import read_yaml
from src.utils.data import download_data

class Ingestion:

    def __init__(self):
        self.config = read_yaml(CONFIG_PATH)
        print(self.config)
        self.ingestion_config = self.config["data_ingestion"]
        self.url = self.ingestion_config["data_url"]
        self.zip_file_name = self.ingestion_config["zip_file_name"]
        self.extract_path = os.path.join(PROJECT_ROOT, self.ingestion_config["extract_path"])
        self.download_path = os.path.join(PROJECT_ROOT, self.ingestion_config["download_path"])

    def process(self):
        """ Downloads and extracts dataset."""

        # downloads and extracts the data
        download_data(
            url = self.url,
            download_path = self.download_path,
            extract_path = self.extract_path,
            file_name = self.zip_file_name
        )

