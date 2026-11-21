from dataclasses import dataclass, field

import numpy as np

from ga_solver.config import GAConfig
from ga_solver.ga.fitness import Func, evaluate_error, fitness_from_error
from ga_solver.ga.individual import init_population
from ga_solver.ga.operators import arithmetic_crossover, gaussian_mutation, tournament_selection


@dataclass
class GAResult:
    best_x: float
    best_error: float
    n_generations: int
    best_history: list[float] = field(default_factory=list)
    mean_history: list[float] = field(default_factory=list)
    final_population: np.ndarray | None = None

    @property
    def converged(self) -> bool:
        return np.isfinite(self.best_error)


def run_ga(f: Func, config: GAConfig) -> GAResult:
    rng = np.random.default_rng(config.seed)
    lo, hi = config.lower_bound, config.upper_bound

    population = init_population(rng, config.pop_size, lo, hi)
    best_history: list[float] = []
    mean_history: list[float] = []
    n_children = config.pop_size - config.elite_size
    best_err_so_far = np.inf
    stall = 0

    for gen in range(1, config.generations + 1):
        err = evaluate_error(f, population)
        fitness = fitness_from_error(err)
        best_history.append(float(fitness.max()))
        mean_history.append(float(fitness.mean()))

        if err.min() < config.tolerance:
            break

        if err.min() < best_err_so_far * (1 - 1e-9):
            best_err_so_far = err.min()
            stall = 0
        else:
            stall += 1
            if config.patience is not None and stall >= config.patience:
                break

        elite_idx = np.argsort(err)[:config.elite_size]
        elites = population[elite_idx]

        n_parents = n_children + (n_children % 2)
        parents = tournament_selection(rng, population, fitness, n_parents, config.tournament_k)
        rng.shuffle(parents)

        children = np.empty(n_parents)
        for i in range(0, n_parents, 2):
            if rng.random() < config.crossover_rate:
                children[i], children[i + 1] = arithmetic_crossover(rng, parents[i], parents[i + 1])
            else:
                children[i], children[i + 1] = parents[i], parents[i + 1]

        children = gaussian_mutation(rng, children[:n_children], config.mutation_rate,
                                     config.mutation_std, lo, hi)

        population = np.concatenate([elites, children])

    err = evaluate_error(f, population)
    best = int(np.argmin(err))
    return GAResult(
        best_x=float(population[best]),
        best_error=float(err[best]),
        n_generations=gen,
        best_history=best_history,
        mean_history=mean_history,
        final_population=population,
    )