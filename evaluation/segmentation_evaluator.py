"""Module to evaluate the final image segmentation, i.e. the prediction by the network. It uses fully labeled images
and the networks prediction in order to output metrics."""
from pathlib import Path
from PIL import Image

from network.input_type import InputType, is_urban_only_rgb_input, is_urban_input
from preprocessing.preprocessing import RawImageProcessor
from preprocessing.hyperspectral_preprocessing import HyperspectralImageProcessor
from evaluation.export_setup import export_prediction
from preprocessing.preprocessing_extensions import get_resize_x_y
from preprocessing.urban_preprocessing import UrbanImageProcessor, UrbanLabelProcessor
from run.run_config import RunConfig
import numpy as np
import cv2
import torch

from run.run_context import RunContext


def load_labels(label_folder, resize_x, resize_y):
    """A function that loads labels from a given folder. It uses segmentation masks (mask in this case
    means white pixels in the image). It concatenates all given labels in channel dimension.

    :return: Returns the labels stacked in channel dimension
    """
    label_folder = Path(label_folder)
    # build labels
    ground_truth = []
    label_nr = 0
    label_mapping = {}
    for file in sorted(label_folder.iterdir()):
        image = np.asarray(Image.open(file))
        image = cv2.resize(image, dsize=(resize_x, resize_y), interpolation=cv2.INTER_NEAREST)
        mask = np.sum(image, axis=2)
        mask = (mask == np.max(mask)).astype(int)

        label_values = mask[..., np.newaxis]
        ground_truth.append(label_values)
        label_name = str(file).split('.')[1]
        label_mapping.update({label_nr: label_name})
        label_nr += 1

    return np.concatenate(ground_truth, axis=2)


def reshape_labels(labels):
    reshaped_labels = torch.from_numpy(labels).permute(2, 0, 1).unsqueeze(0)
    reshaped_labels = torch.argmax(reshaped_labels, dim=1)
    return reshaped_labels


def evaluate_other_take(model, resize_factor, device, export_path, take_name, only_rgb):
    take_img = RawImageProcessor(resize_factor=resize_factor, only_rgb=only_rgb).preprocess_folder(
        f'takes/{take_name}/channels/')
    device_data = take_img.float().to(device)
    predicted = model(device_data)
    export_prediction(f'{export_path}/{take_name}.png', predicted)


class SegmentationEvaluator:
    def __init__(self, model, model_fingerprint, run_config: RunConfig, run_context: RunContext,
                 loss, logger):
        # self.labels = load_labels(label_folder, resize_x, resize_y)
        # self.labels_2D = reshape_labels(self.labels)
        self.model = model
        self.model_fingerprint = model_fingerprint
        self.run_config = run_config
        self.run_context = run_context
        self.loss = loss
        self.logger = logger

    def evaluate_cnn(self):
        self.logger.info(f'=== EVALUATION Run ===')
        self.logger.info(f'Last loss: {self.loss.item()}')

        if len(self.run_config.eval_takes) == 0:
            self.logger.warn('No takes to evaluate!')
            return None

        accuracies = []
        non_bg_accuracies = []

        for take_name in self.run_config.eval_takes:

            raw_img_data = None
            labels_2d = None
            if take_name in self.run_context.evaluation_data_cache.take_samples:
                raw_img_data, labels_2d = self.run_context.evaluation_data_cache.get_take(take_name)
                self.logger.info('Loaded cached data')
            else:
                input_type = self.run_config.input_type
                if input_type == InputType.REAL_HYPERSPECTRAL or input_type == InputType.REAL_HYPER_ONLY_RGB :
                    processor = HyperspectralImageProcessor(self.run_config)
                    raw_img_data = processor.load_take(take_name)
                if input_type == InputType.FRUITS_YELLOW or input_type == InputType.FRUITS_YELLOW_ONLY_RGB :
                    processor = HyperspectralImageProcessor(self.run_config)
                    raw_img_data = processor.load_take(take_name)
                if is_urban_input(input_type) or is_urban_only_rgb_input(input_type):
                    raw_img_data = UrbanImageProcessor(self.run_config).load_take(take_name)
                if input_type == InputType.SIM_HYPER_ONLY_RGB or input_type == InputType.SIM_HYPERSPECTRAL:
                    data_folder = f'takes/{take_name}/channels'
                    raw_img_data = RawImageProcessor(self.run_config).preprocess_folder(data_folder)

                if is_urban_input(input_type) or is_urban_only_rgb_input(input_type):
                    labels = UrbanLabelProcessor(self.run_config).load_labels_from_take_percentage(take_name)
                    labels_2d = torch.argmax(labels, dim=1)
                else:
                    # labeling is same for both, data is different
                    label_folder = f'takes/{take_name}/labeling'
                    resize_x = raw_img_data.shape[3]
                    resize_y = raw_img_data.shape[2]
                    labels = load_labels(label_folder, resize_x, resize_y)
                    labels_2d = reshape_labels(labels)

                raw_img_data = raw_img_data.to(self.run_config.device).float()
                # store data for subsequent runs
                self.run_context.evaluation_data_cache.update_take(take_name, raw_img_data, labels_2d)

            prediction = self.model(raw_img_data)
            overall_accuracy = self._accuracy_cnn(prediction, labels_2d)
            non_bg_accuracy = self._accuracy_cnn_only_non_bg(prediction, labels_2d)

            img_name = f'{self.run_config.export_folder}/{self.run_config.fingerprint}/{self.model_fingerprint}_take_{take_name}.png'
            export_prediction(img_name, prediction)

            accuracies.append(overall_accuracy)
            non_bg_accuracies.append(non_bg_accuracy)
            self._write_evaluation(take_name, overall_accuracy, non_bg_accuracy)

        avg_accuracy, avg_non_bg_accuracy = self._write_average_metrics(accuracies, non_bg_accuracies)
        evaluations = {
            'accuracy': avg_accuracy,
            'accuracy_non_bg': avg_non_bg_accuracy
        }
        return evaluations

    def _write_average_metrics(self, accuracies, non_bg_accuracies):
        avg_accuracy = np.mean(accuracies)
        avg_non_bg_accuracy = np.mean(non_bg_accuracies)
        self.logger.info(f'Average:')
        self.logger.info(f'\t\tAverage overall accuracy: {avg_accuracy}')
        self.logger.info(f'\t\tAverage non background accuracy: {avg_non_bg_accuracy}')
        return avg_accuracy, avg_non_bg_accuracy

    def evaluate_fully_connected(self, prediction):
        raise NotImplementedError('Deprecated!')
        # get class for pixel and reshape to image format
        result = torch.argmax(prediction, dim=1).cpu().detach().numpy()
        result_reshaped = torch.unsqueeze(torch.from_numpy(result.reshape(self.resize_y, self.resize_x)), 0)
        # calculate accuracies
        overall_accuracy = self._accuracy_fully_connected(result_reshaped)
        non_bg_accuracy = self._accuracy_fully_connected_only_non_bg(result_reshaped)
        self._write_evaluation(overall_accuracy, non_bg_accuracy)

    def _accuracy_cnn(self, prediction, labels_2d):
        result = torch.argmax(prediction, dim=1).cpu().detach().numpy()
        total_pixels = labels_2d.shape[1] * labels_2d.shape[2]
        return np.equal(labels_2d, result).sum().item() / total_pixels

    def _accuracy_cnn_only_non_bg(self, prediction, labels_2d):
        result = torch.argmax(prediction, dim=1).cpu().detach().numpy()
        non_bg_pixels = labels_2d[labels_2d != 0]
        prediction_non_bg = result[labels_2d != 0]
        return np.equal(non_bg_pixels, prediction_non_bg).sum().item() / len(non_bg_pixels)

    def _accuracy_fully_connected(self, prediction):
        raise NotImplementedError('Deprecated!')
        return np.equal(self.labels_2D, prediction).sum().item() / self.total_pixels

    def _accuracy_fully_connected_only_non_bg(self, prediction):
        raise NotImplementedError('Deprecated!')
        non_bg_pixels = self.labels_2D[self.labels_2D != 0]
        result_non_bg = prediction[self.labels_2D != 0]
        return np.equal(non_bg_pixels, result_non_bg).sum().item() / len(non_bg_pixels)

    def _write_evaluation(self, take_nr, overall_accuracy, non_bg_accuracy):
        self.logger.info(f'Take No. {take_nr}')
        self.logger.info(f'\t\tOverall accuracy: {overall_accuracy}')
        self.logger.info(f'\t\tAccuracy only non background: {non_bg_accuracy}')
