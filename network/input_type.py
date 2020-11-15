from enum import Enum

"""Helper module for input types used in this project"""


def is_only_rgb(in_type):
    if in_type == InputType.URBAN_HYPERSPECTRAL_4_ONLY_RGB:
        return True
    if in_type == InputType.URBAN_HYPERSPECTRAL_5_ONLY_RGB:
        return True
    if in_type == InputType.URBAN_HYPERSPECTRAL_6_ONLY_RGB:
        return True
    if in_type == InputType.REAL_HYPER_ONLY_RGB:
        return True
    if in_type == InputType.SIM_HYPER_ONLY_RGB:
        return True
    return False


def is_urban_input(in_type):
    if in_type == InputType.URBAN_HYPERSPECTRAL_4:
        return True
    if in_type == InputType.URBAN_HYPERSPECTRAL_5:
        return True
    if in_type == InputType.URBAN_HYPERSPECTRAL_6:
        return True
    return False


def is_urban_only_rgb_input(in_type):
    if in_type == InputType.URBAN_HYPERSPECTRAL_4_ONLY_RGB:
        return True
    if in_type == InputType.URBAN_HYPERSPECTRAL_5_ONLY_RGB:
        return True
    if in_type == InputType.URBAN_HYPERSPECTRAL_6_ONLY_RGB:
        return True
    return False


def get_dimensions(in_type):
    if is_only_rgb(in_type):
        return 3
    if in_type == InputType.SIM_HYPERSPECTRAL:
        return 39
    if in_type == InputType.REAL_HYPERSPECTRAL:
        return 49
    if is_urban_input(in_type):
        return 162

    raise ValueError('No valid input type specified!')


def get_class_number(in_type):
    if in_type == InputType.SIM_HYPER_ONLY_RGB or in_type == InputType.SIM_HYPERSPECTRAL:
        return 15
    if in_type == InputType.REAL_HYPERSPECTRAL or in_type == InputType.REAL_HYPER_ONLY_RGB:
        return 3
    if in_type == InputType.URBAN_HYPERSPECTRAL_4 or in_type == InputType.URBAN_HYPERSPECTRAL_4_ONLY_RGB:
        return 4
    if in_type == InputType.URBAN_HYPERSPECTRAL_5 or in_type == InputType.URBAN_HYPERSPECTRAL_5_ONLY_RGB:
        return 5
    if in_type == InputType.URBAN_HYPERSPECTRAL_6 or in_type == InputType.URBAN_HYPERSPECTRAL_6_ONLY_RGB:
        return 6
    raise ValueError('No valid input type specified!')


def get_loss_indices(in_type):
    if in_type == InputType.SIM_HYPER_ONLY_RGB or in_type == InputType.SIM_HYPERSPECTRAL:
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    if in_type == InputType.REAL_HYPERSPECTRAL or in_type == InputType.REAL_HYPER_ONLY_RGB:
        return [0, 1, 2]
    if in_type == InputType.URBAN_HYPERSPECTRAL_4 or in_type == InputType.URBAN_HYPERSPECTRAL_4_ONLY_RGB:
        return [0, 1, 2, 3]
    if in_type == InputType.URBAN_HYPERSPECTRAL_5 or in_type == InputType.URBAN_HYPERSPECTRAL_5_ONLY_RGB:
        return [0, 1, 2, 3, 4]
    if in_type == InputType.URBAN_HYPERSPECTRAL_6 or in_type == InputType.URBAN_HYPERSPECTRAL_6_ONLY_RGB:
        return [0, 1, 2, 3, 4, 5]

    raise ValueError('No valid input type specified!')


class InputType(Enum):
    SIM_HYPER_ONLY_RGB = 1
    SIM_HYPERSPECTRAL = 2
    REAL_HYPERSPECTRAL = 3
    REAL_HYPER_ONLY_RGB = 4
    URBAN_HYPERSPECTRAL_4 = 5
    URBAN_HYPERSPECTRAL_4_ONLY_RGB = 6
    URBAN_HYPERSPECTRAL_5 = 7
    URBAN_HYPERSPECTRAL_5_ONLY_RGB = 8
    URBAN_HYPERSPECTRAL_6 = 9
    URBAN_HYPERSPECTRAL_6_ONLY_RGB = 10
