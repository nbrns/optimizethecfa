"""Module to execute runs with the Runner class. The class requires a RunConfig instance."""
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from network.input_type import InputType, is_urban_input, is_urban_only_rgb_input, get_dimensions
from preprocessing.urban_preprocessing import UrbanImageProcessor, UrbanLabelProcessor
from run.run_context import RunContext
from preprocessing.preprocessing import RawImageProcessor, LabelBuilder
from preprocessing.hyperspectral_preprocessing import HyperspectralImageProcessor
from evaluation.export_setup import export_setup
from evaluation.segmentation_evaluator import SegmentationEvaluator
from evaluation.final_run_evaluator import FinalRunEvaluator
from evaluation.training_scribble_evaluator import TrainingScribbleEvaluator
from network.custom_loss import METVCVLoss
from datetime import datetime
from network.network_builder import get_network


class Runner:
    def __init__(self, run_config, run_amount, eval_best_amount):
        self.run_config = run_config
        self.run_context = RunContext()
        self.run_amount = run_amount
        self.eval_best_amount = eval_best_amount
        self.run_counter = 0

        # CONFIGURE LOGGING
        self.export_path = Path.joinpath(Path(self.run_config.export_folder), Path(self.run_config.fingerprint))
        Path.mkdir(self.export_path, exist_ok=True, parents=True)
        formatting = '%(asctime)s - %(name)s | %(message)s'
        logging.basicConfig(level='INFO', format=formatting, handlers=[
            logging.FileHandler(f'{self.export_path}/run.log'),
            logging.StreamHandler(sys.stdout)
        ])

    def _run(self):
        model = get_network(self.run_config).to(self.run_config.device)
        model_name = model.__class__.__name__
        model_fingerprint = f'{model_name}_{datetime.now().strftime("%m%d%Y-%H%M%S")}'

        run_logger = logging.getLogger(f'run-{self.run_counter + 1}')
        eval_logger = logging.getLogger(f'evaluation-{self.run_counter + 1}')

        optimizer = self.run_config.optimizer(model.parameters(), lr=self.run_config.learning_rate)

        loss_function = METVCVLoss(run_config=self.run_config)
        run_logger.info('Processing data...')

        img = None
        labels = None
        if not self.run_context.run_data_cache.has_cached_values():
            input_type = self.run_config.input_type
            if get_dimensions(input_type) == 49 or input_type == InputType.REAL_HYPER_ONLY_RGB:
                processor = HyperspectralImageProcessor(self.run_config)
                img = processor.load_takes()
                labels, label_mapping = processor.load_labels_from_takes(self.run_config.takes)
            if is_urban_input(input_type) or is_urban_only_rgb_input(input_type):
                img = UrbanImageProcessor(self.run_config).load_takes()
                labels, label_mapping = UrbanLabelProcessor(self.run_config)\
                    .load_labels_from_take(self.run_config.takes)
            if input_type == InputType.SIM_HYPERSPECTRAL or input_type == InputType.SIM_HYPER_ONLY_RGB:
                img = RawImageProcessor(self.run_config).load_takes()
                labels, label_mapping = LabelBuilder(self.run_config)\
                    .load_labels_from_take(takes=self.run_config.takes, verbose=self.run_config.verbose)

            run_logger.info(f'First image is of shape {img[0].shape}')
            run_logger.info(f'First label is of shape {labels[0].shape}')

            # store processed data for subsequent runs
            self.run_context.run_data_cache.set_data(img, labels, label_mapping)
            run_logger.info('Stored data in cache')

        else:
            run_logger.info('Using cached samples and labels')
            img, labels, label_mapping = self.run_context.run_data_cache.get_data()

        if self.run_counter == 0:
            if self.run_config.verbose:
                export_setup(model, loss_function, label_mapping, run_logger)
            run_logger.info(self.run_config)

        run_logger.info('Training...')
        data_torch = [s.float().to(self.run_config.device) for s in img]
        labels_torch = [l.float().to(self.run_config.device) for l in labels]

        # Training
        losses = []
        for epoch in range(self.run_config.epochs):
            for sample, label in zip(data_torch, labels_torch):
                y_pred = model(sample)
                loss = loss_function(y_pred, label, model.cfa_sim_layer.conv.weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                try:
                    model.cfa_sim_layer.clamp()
                except AttributeError:
                    # try execute, no error if does not exist
                    pass

            if epoch % 100 == 0:
                if self.run_config.verbose:
                    classes = np.unique(y_pred.argmax(dim=1).cpu().detach())
                    print(classes)
                run_logger.info((epoch, loss.item()))
                losses.append((epoch, loss.item()))

        # Evaluation
        scribble_evaluator = TrainingScribbleEvaluator(model, data_torch, labels_torch, eval_logger)
        training_results = scribble_evaluator.evaluate_scribbles()
        evaluator = SegmentationEvaluator(model=model,
                                          model_fingerprint=model_fingerprint,
                                          run_config=self.run_config,
                                          run_context=self.run_context,
                                          loss=loss,
                                          logger=eval_logger
                                          )
        evaluation_results = evaluator.evaluate_cnn()
        evaluation_results.update(training_results)

        for w in model.cfa_sim_layer.conv.weight:
            plt.plot(w.squeeze().detach().cpu(), linewidth=1)
        plt.savefig(f'{self.export_path}/cfa_weights_{model_fingerprint}.png')
        plt.clf()
        return evaluation_results

    def execute_runs(self):
        # logging
        logger = logging.getLogger('main')
        logger.info(f'CFA Out: {self.run_config.cfa_sim_out_channels}')
        final_eval_logger = logging.getLogger(f'final_evaluation-{self.run_counter + 1}')
        final_eval_logger.addHandler(logging.FileHandler(f'{self.export_path}/eval.log'))
        metrics_logger = logging.getLogger('metrics')
        metrics_logger.handlers = []
        metrics_logger.addHandler(logging.FileHandler(f'{self.export_path}/metrics.csv'))

        # execute
        evaluations = []
        for i in range(self.run_amount):
            # torch.cuda.empty_cache()
            logger.info(f'--- Run No. {i + 1} ---')
            evaluation = self._run()
            evaluations.append(evaluation)
            self.run_counter += 1

        # evaluate
        FinalRunEvaluator(evaluations,
                          self.run_config,
                          self.run_amount,
                          self.eval_best_amount,
                          final_eval_logger,
                          metrics_logger).evaluate()
