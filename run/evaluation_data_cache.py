"""Simple cache for evaluation data, accelerates multiple run execution"""
class EvaluationDataCache:
    def __init__(self):
        self.take_samples = {}
        self.take_labels = {}

    def get_take(self, take_nr):
        return self.take_samples[take_nr], self.take_labels[take_nr]

    def update_take(self, take_nr, sample, label):
        self.take_samples.update({take_nr: sample})
        self.take_labels.update({take_nr: label})
