import os
from core.config import RESULTS_DIR

MAIN_RESULTS_DIR = os.path.join(RESULTS_DIR, "main")
OVERRIDING_RESULTS_DIR = os.path.join(RESULTS_DIR, "overriding")


def main_experiment_results_dir(experiment_id: str) -> str:
    return os.path.join(MAIN_RESULTS_DIR, experiment_id)


def overriding_experiment_results_dir(experiment_id: str) -> str:
    return os.path.join(OVERRIDING_RESULTS_DIR, experiment_id)
