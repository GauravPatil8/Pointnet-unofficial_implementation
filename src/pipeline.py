from src.components.Ingestion import Ingestion
from src.components.Trainer import Trainer

class Pipeline:
    def __init__(self):
        self._stages = [Ingestion, Trainer]

    def run_pipeline(self):
        for stage in self.stages:
            stage_init = stage()
            stage_init.process()