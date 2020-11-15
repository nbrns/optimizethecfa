"""Module to store run configurations. A run configuration is represented by the RunConfig class."""
from datetime import datetime


class RunConfig:
    def __init__(self, device, network_identifier, epochs, fully_supervised, optimizer, learning_rate, resize_factor,
                 tvl_alpha, cvl_beta, export_folder, takes, eval_takes, input_type, bayer_pattern_size, cfa_sim_out_channels,
                 correct_pixel_shift, balance_labels, label_percentage=1.0, idc=None, verbose=False) -> object:
        """
        Initializes a RunConfig instance
        :param device: The device to run on, e.g. 'cuda:0'
        :param network_identifier: Specifies the network architecture (see :mod:`network.network_builder`)
        :param epochs: Epochs for training
        :param fully_supervised: Use full labels or scribbles
        :param optimizer: Requires an optimizer class to use
        :param learning_rate: Learning rate for the optimizer
        :param resize_factor: Factor to resize the initial take images
        :param tvl_alpha: alpha parameter for TVL weighting
        :param cvl_beta: beta parameter cor CVL weighting
        :param export_folder: Folder to export the run logs and predicitons to
        :param takes: List of takes to use for training, e.g. [1, 2, 4]. Specified by naming conventions of takes
            in takes folder
        :param eval_takes: List of takes to use for evaluation
        :param input_type: Enumerator of input type
        :param bayer_pattern_size: Size of bayer pattern kernel
        :param cfa_sim_out_channels: Output channels of CFA simulation layer, i.e. number of learnable colors
        :param correct_pixel_shift: Correct pixel shifting of bayer kernels
        :param balance_labels: Correct class imbalances
        :param label_percentage: Percentage of labels to use, only for urban input (requires data loader adaption)
        :param idc: Specifies classes to be valued in loss function
        :param verbose: Verbose output of Runner
        """
        self.device = device
        self.network_identifier = network_identifier
        self.epochs = epochs
        self.fully_supervised = fully_supervised
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.resize_factor = resize_factor
        self.tvl_alpha = tvl_alpha
        self.cvl_beta = cvl_beta
        self.export_folder = export_folder
        self.takes = takes
        self.eval_takes = eval_takes
        self.input_type = input_type
        self.bayer_pattern_size = bayer_pattern_size
        self.cfa_sim_out_channels = cfa_sim_out_channels
        self.correct_pixel_shift = correct_pixel_shift
        self.balance_labels = balance_labels
        self.label_percentage = label_percentage
        self.idc = idc
        self.verbose = verbose

        self.fingerprint = ''
        self.update_fingerprint()

    def __get_model_name(self):
        model_name = str(self.network_identifier).split('.')[1]
        return model_name

    def update_fingerprint(self):
        self.fingerprint = f'{self.__get_model_name()}_{datetime.now().strftime("%m%d%Y-%H%M%S")}'

    def __str__(self):
        representation = 'Run Config:\n'
        for name, value in vars(self).items():
            representation += f'\t{name}: {value}\n'
        return representation
