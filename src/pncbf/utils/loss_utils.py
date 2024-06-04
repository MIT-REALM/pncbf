from loguru import logger

from pncbf.utils.jax_types import FloatScalar


def weighted_sum_dict(loss_dict: dict[str, FloatScalar], weights_dict: dict[str, FloatScalar]) -> FloatScalar:
    # 1: Make sure all keys in weights_dict are in loss_dict.
    total_loss = 0
    for k, weight in weights_dict.items():
        if k not in loss_dict:
            logger.warning("Weight dict key {} not in loss dict! {}".format(k, loss_dict.keys()))
            continue

        total_loss = total_loss + weight * loss_dict[k]

    return total_loss
