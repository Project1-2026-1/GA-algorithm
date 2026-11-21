import numpy as np


def tournament_selection(rng: np.random.Generator, pop: np.ndarray,
                         fitness: np.ndarray, n_select: int, k: int = 3) -> np.ndarray:
    k = min(k, len(pop))
    candidates = np.argsort(rng.random((n_select, len(pop))), axis=1)[:, :k]
    winners = candidates[np.arange(n_select), np.argmax(fitness[candidates], axis=1)]
    return pop[winners]


def arithmetic_crossover(rng: np.random.Generator, p1: float, p2: float) -> tuple[float, float]:
    alpha = rng.random()
    return alpha * p1 + (1 - alpha) * p2, alpha * p2 + (1 - alpha) * p1


def gaussian_mutation(rng: np.random.Generator, pop: np.ndarray, rate: float,
                      std: float, lower: float, upper: float) -> np.ndarray:
    pop = pop.copy()
    mask = rng.random(len(pop)) < rate
    pop[mask] += rng.normal(0, std, mask.sum())
    return np.clip(pop, lower, upper)