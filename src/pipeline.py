import logging
from src.components.Ingestion import Ingestion
from src.components.Trainer import Trainer

class Pipeline:
    def __init__(self):
        self._stages = [Ingestion, Trainer]
        self.logger = logging.getLogger(__name__)

    def run_pipeline(self):
        for stage in self.stages:
            self.logger(stage)
            stage_init = stage()
            stage_init.process()