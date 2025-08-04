import os
import logging
from src.pipeline import run_pipeline

if __name__ == "__main__":
    os.makedirs(R"src/logs/", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(R"src/logs/info.log"),
            logging.StreamHandler()
        ]
    )
    run_pipeline()