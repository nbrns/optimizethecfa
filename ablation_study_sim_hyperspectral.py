"""
Module to run multiple trainings of networks for cfa optimization. Runs are configured in this module. May also be considered
as some kind of main script.
"""
import torch.nn

from network.input_type import InputType
from run.run_config import RunConfig
from run.runner import Runner
from network.network_builder import NetworkIdentifier

desired_gpu_nr = 0
device = torch.device(f'cuda:{desired_gpu_nr}' if torch.cuda.is_available() else "cpu")
run_config = RunConfig(
    device=device,
    network_identifier=NetworkIdentifier.REDUCED_UNET,
    epochs=5000,
    fully_supervised=False,
    optimizer=torch.optim.Adam,
    learning_rate=1e-3,
    resize_factor=0.25,
    tvl_alpha=10,
    export_folder='ablation_study_urban/',
    takes=[1, 2, 4],
    eval_takes=[1, 2, 4],
    input_type=InputType.SIM_HYPERSPECTRAL,
    bayer_pattern_size=2,
    cfa_sim_out_channels=3,
    correct_pixel_shift=True,
    balance_labels=False,
    label_percentage=1.0,
    verbose=False
)

run_amount = 10
eval_best_amount = 3

networks = [
    NetworkIdentifier.REDUCED_UNET,
    NetworkIdentifier.BAYER_UNET,
    NetworkIdentifier.BAYER_DENSENET,
    NetworkIdentifier.CONV_40_1,
    NetworkIdentifier.CONV_40_3,
    NetworkIdentifier.CONV_40_5,
    NetworkIdentifier.CONV_40_7,
    NetworkIdentifier.CONV_40_9
]

input_types = [
    InputType.SIM_HYPERSPECTRAL,
    InputType.SIM_HYPER_ONLY_RGB
]

for network in networks:
    for input_type in input_types:
        run_config.input_type = input_type
        run_config.network_identifier = network
        # use adagrad from conv_40_7
        if run_config.input_type == NetworkIdentifier.CONV_40_7:
            run_config.optimizer = torch.optim.adagrad
        run_config.update_fingerprint()
        Runner(run_config, run_amount, eval_best_amount).execute_runs()
