import logging
from src.components.Ingestion import Ingestion
from src.components.Trainer import Trainer

class Pipeline:
    def __init__(self):
        self._stages = [Ingestion, Trainer]
        self.logger = logging.getLogger(__name__)

    def run_pipeline(self):
        for stage in self._stages:
            stage_init = stage()
            self.logger.info(stage_init)
            stage_init.process()