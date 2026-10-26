"""Environment appearance changes; these are not neural interventions."""

from copy import deepcopy


def appearance_config(environment_config, colour_mode):
    if colour_mode not in {"correlated", "randomised", "reversed"}:
        raise ValueError("Invalid colour mode")
    result = deepcopy(environment_config)
    result["colour_mode"] = colour_mode
    return result
