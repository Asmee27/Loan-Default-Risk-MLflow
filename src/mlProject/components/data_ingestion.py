import os
import shutil
from mlProject import logger
from mlProject.utils.common import get_size
from mlProject.entity.config_entity import DataIngestionConfig
from pathlib import Path


class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        """
        Since the dataset is already downloaded locally,
        this method verifies that the CSV exists.
        """

        if os.path.exists(self.config.local_data_file):
            logger.info(
                f"Dataset already exists: {self.config.local_data_file}"
            )
            logger.info(
                f"Dataset size: {get_size(Path(self.config.local_data_file))}"
            )
        else:
            raise FileNotFoundError(
                f"Dataset not found at: {self.config.local_data_file}"
            )

    def extract_zip_file(self):
        """
        No extraction is required because the dataset
        is already available as a CSV file.
        """

        logger.info("Dataset is already in CSV format. No extraction required.")