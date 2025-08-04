from src.components.Ingestion import Ingestion
stages = [Ingestion]

def run_pipeline():
    for stage in stages:
        stage_init = Ingestion()
        stage_init.process()