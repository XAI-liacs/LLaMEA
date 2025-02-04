import numpy as np


def score(item, bins):
    scores = (
        (bins / np.sqrt(np.log(bins - item))) ** (bins / np.sqrt(item))
        * np.exp(item * (bins - item))
        * np.sqrt(item)
    )
    scores /= (1 / bins) * np.sqrt(item)
    scores *= 100  # scaler constant factor
    return scores
