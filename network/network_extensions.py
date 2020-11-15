from run.run_config import RunConfig


def get_input_dimensions_run_config(run_config: RunConfig):
    """Returns input dimensions of network depending on run config (only RGB, hyperspectral image, ...)."""
    d_in = 39
    if run_config.hyperspectral:
        d_in = 49
    if run_config.only_rgb:
        d_in = 3

    return d_in


def get_input_dimensions_parameters(only_rgb, hyperspectral):
    """Returns input dimensions of network depending on passed parameters."""
    d_in = 39
    if hyperspectral:
        d_in = 49
    if only_rgb:
        d_in = 3

    return d_in


