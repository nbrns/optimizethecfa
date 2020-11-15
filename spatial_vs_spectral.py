"""
Module to run multiple trainings of networks for cfa optimization. Runs are configured in this module.
May also be considered as some kind of main script.
"""
import torch.nn
from network.input_type import InputType
from run.run_config import RunConfig
from run.runner import Runner
from network.network_builder import NetworkIdentifier

desired_gpu_nr = 3
device = torch.device(f'cuda:{desired_gpu_nr}' if torch.cuda.is_available() else "cpu")
run_config = RunConfig(
    device=device,
    network_identifier=NetworkIdentifier.CONV_40_1,
    epochs=5000,
    fully_supervised=False,
    optimizer=torch.optim.Adam,
    learning_rate=1e-3,
    resize_factor=0,
    tvl_alpha=0,
    export_folder='spatial_vs_spectral/',
    takes=['u4c'],
    eval_takes=['u4c'],
    input_type=InputType.URBAN_HYPERSPECTRAL_4,
    bayer_pattern_size=2,
    cfa_sim_softmax=False,
    cfa_sim_out_channels=3,
    correct_pixel_shift=True,
    balance_labels=False,
    label_percentage=1.0,
    verbose=False
)

run_amount = 6
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

for network in networks:
    run_config.network_identifier = network
    run_config.bayer_pattern_size = 2
    for i in range(1, 10):
        print(f'CFA out: {i}')
        run_config.cfa_sim_out_channels = i
        if i == 5:
            run_config.bayer_pattern_size = 3
        run_config.update_fingerprint()
        Runner(run_config, run_amount, eval_best_amount).execute_runs()
