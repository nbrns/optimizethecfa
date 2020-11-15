import numpy as np


class FinalRunEvaluator:
    def __init__(self, evaluation_results, run_config, run_amount, best_run_amount, logger, metrics_logger=None):
        self.logger = logger
        self.evaluation_results = evaluation_results
        self.metrics_logger = metrics_logger
        self.best_run_amount = best_run_amount
        self.run_amount = run_amount
        self.run_config = run_config

    def evaluate(self):
        sorted_evals = sorted(self.evaluation_results, key=lambda x: x['training_accuracy'], reverse=True)

        training_acc = [e['training_accuracy'] for e in sorted_evals]
        acc = [e['accuracy'] for e in sorted_evals]
        non_bg_acc = [e['accuracy_non_bg'] for e in sorted_evals]

        overall_training_acc = np.mean(training_acc)
        overall_acc = np.mean(acc)
        overall_non_bg_acc = np.mean(non_bg_acc)

        best_training_values = training_acc[:self.best_run_amount]
        best_acc_values = acc[:self.best_run_amount]
        best_non_bg_acc_values = non_bg_acc[:self.best_run_amount]

        mean_best_training_values = np.mean(best_training_values)
        mean_best_acc_values = np.mean(best_acc_values)
        mean_best_non_bg_acc_values = np.mean(best_non_bg_acc_values)

        var_best_training_values = np.var(best_training_values)
        var_best_acc_values = np.var(best_acc_values)
        var_best_non_bg_acc_values = np.var(best_non_bg_acc_values)

        top_training_value = training_acc[0]
        top_acc_value = acc[0]
        top_non_bg_acc_value = non_bg_acc[0]

        if self.metrics_logger is not None:
            self.metrics_logger.info(
                f'{mean_best_training_values},{mean_best_non_bg_acc_values},{mean_best_acc_values}')

        self.logger.info(f'Mean of {self.run_amount} runs:')
        self.logger.info(f'\tModel: {self.run_config.fingerprint} - {self.run_config.input_type}')
        self.logger.info(f'\t\tOverall training accuracy: {overall_training_acc}')
        self.logger.info(f'\t\tOverall accuracy: {overall_acc}')
        self.logger.info(f'\t\tOverall non background accuracy: {overall_non_bg_acc}\n')
        self.logger.info(f'\tTop metrics (according to training accuracy)')
        self.logger.info(f'\t\tTop {self.best_run_amount} training accuracy: {mean_best_training_values}')
        self.logger.info(f'\t\t\tVariance: {var_best_training_values}')
        self.logger.info(f'\t\t\tValues: {best_training_values}')
        self.logger.info(f'\t\tTop {self.best_run_amount} accuracy: {mean_best_acc_values}')
        self.logger.info(f'\t\t\tVariance: {var_best_acc_values}')
        self.logger.info(f'\t\t\tValues: {best_acc_values}')
        self.logger.info(
            f'\t\tTop {self.best_run_amount} non background accuracy: {mean_best_non_bg_acc_values}')
        self.logger.info(f'\t\t\tVariance: {var_best_non_bg_acc_values}')
        self.logger.info(f'\t\t\tValues: {best_non_bg_acc_values}\n')
        self.logger.info(f'\t\tBest training accuracy: {top_training_value}')
        self.logger.info(f'\t\tBest accuracy: {top_acc_value}')
        self.logger.info(f'\t\tBest non background accuracy: {top_non_bg_acc_value}\n')
