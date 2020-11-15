"""
Module to run training of networks for cfa optimization. Runs are configured in this module. May also be considered
as some kind of main script.
"""
import torch.nn
from run.run_config import RunConfig
from run.runner import Runner
from network.network_builder import NetworkIdentifier
from network.input_type import InputType

desired_gpu_nr = 0
device = torch.device(f'cuda:{desired_gpu_nr}' if torch.cuda.is_available() else "cpu")
run_config = RunConfig(
    device=device,
    network_identifier=NetworkIdentifier.BAYER_DENSENET,
    epochs=1000,
    fully_supervised=False,
    optimizer=torch.optim.Adam,
    learning_rate=1e-3,
    resize_factor=0,
    tvl_alpha=0,
    cvl_beta=1,
    export_folder='export/',
    takes=['u4c'],
    eval_takes=['u4c'],
    input_type=InputType.URBAN_HYPERSPECTRAL_4,
    bayer_pattern_size=2,
    cfa_sim_out_channels=4,
    correct_pixel_shift=True,
    balance_labels=True,
    label_percentage=1.0,
    idc=None,
    verbose=True
)

run_amount = 4
eval_best_amount = 2
runner = Runner(run_config, run_amount, eval_best_amount)
runner.execute_runs()

run_config.input_type=InputType.URBAN_HYPERSPECTRAL_4_ONLY_RGB
run_config.update_fingerprint()
runner = Runner(run_config, run_amount, eval_best_amount)
runner.execute_runs()
