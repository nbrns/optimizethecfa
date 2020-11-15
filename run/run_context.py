"""Module to store the run context, e.g. caches."""
from run.evaluation_data_cache import EvaluationDataCache
from run.run_data_cache import RunDataCache


class RunContext:

    def __init__(self):
        self.run_data_cache = RunDataCache()
        self.evaluation_data_cache = EvaluationDataCache()
