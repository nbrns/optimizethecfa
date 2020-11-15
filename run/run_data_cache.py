"""Naive cache implementation for data caching over multiple runs."""

class RunDataCache:

    def __init__(self):
        self.samples = None
        self.labels = None
        self.label_mapping = None

    def has_cached_values(self):
        if self.samples is None or self.labels is None or self.label_mapping is None:
            return False
        return True

    def set_data(self, samples, labels, label_mapping):
        self.samples = samples
        self.labels = labels
        self.label_mapping = label_mapping

    def get_data(self):
        return self.samples, self.labels, self.label_mapping
