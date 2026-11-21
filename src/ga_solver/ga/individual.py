import numpy as np


def init_population(rng: np.random.Generator, size: int,
                    lower: float, upper: float) -> np.ndarray:
    return rng.uniform(lower, upper, size)