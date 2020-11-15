"""Module to evaluate the training accuracy, i.e. the accuracy only on the scribbled data. For both, prediciton and
labeling, only the scribbled pixels will be considered."""
import numpy as np


class TrainingScribbleEvaluator:
    def __init__(self, model, samples, labels, logger):
        self.model = model
        self.samples = samples
        self.labels = labels
        self.logger = logger
        self.throwaway_class = self.labels[0].shape[1]-1

    def evaluate_scribbles(self):
        self.logger.info('=== TRAINING EVALUATION ===')
        predictions = [self.model(s) for s in self.samples]
        labels = self.labels

        ratios = []
        for prediction, label in zip(predictions, labels):
            prediction_arg_max = prediction.argmax(dim=1)
            label_arg_max = label.argmax(dim=1)
            scribbles = np.where(label_arg_max.cpu() != self.throwaway_class)
            scribbles_pred = prediction_arg_max[scribbles]
            scribbles_labels = label_arg_max[scribbles]
            correct = np.where(scribbles_labels.cpu() == scribbles_pred.cpu())
            correct_ratio = len(correct[0]) / len(scribbles[0])
            ratios.append(correct_ratio)

        total_correct_ratio = np.mean(ratios)
        self.logger.info(f'Training accuracy (mean over samples): {total_correct_ratio}')

        training_eval_results = {
            'training_accuracy': total_correct_ratio
        }
        return training_eval_results
