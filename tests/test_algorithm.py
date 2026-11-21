import numpy as np
import pytest

from ga_solver.config import GAConfig
from ga_solver.ga.algorithm import run_ga
from ga_solver.ga.fitness import evaluate_error, fitness_from_error


def test_error_handles_domain_errors():
    err = evaluate_error(np.log, np.array([-1.0, 0.0, 1.0]))
    assert np.isinf(err[0]) and np.isinf(err[1]) and err[2] == 0.0


def test_error_handles_overflow():
    err = evaluate_error(lambda x: np.exp(5 * x), np.array([1e10]))
    assert np.isinf(err[0])


def test_fitness_range():
    fit = fitness_from_error(np.array([0.0, 1.0, np.inf]))
    assert np.allclose(fit, [1.0, 0.5, 0.0])


def test_linear_equation():
    res = run_ga(lambda x: x - 3, GAConfig(seed=1))
    assert res.best_x == pytest.approx(3, abs=1e-4)


def test_quadratic_positive_root():
    res = run_ga(lambda x: x**2 - 4, GAConfig(lower_bound=0, upper_bound=10, seed=1))
    assert res.best_x == pytest.approx(2, abs=1e-4)


def test_exponential_equation():
    f = lambda x: np.exp(5 * x) - 7 * x * 10 + 6 * x * 4 - 102
    res = run_ga(f, GAConfig(lower_bound=0, upper_bound=3, seed=42))
    assert abs(f(res.best_x)) < 1e-4


def test_reproducible_with_seed():
    cfg = GAConfig(seed=123)
    a = run_ga(lambda x: np.sin(x), cfg)
    b = run_ga(lambda x: np.sin(x), cfg)
    assert a.best_x == b.best_x and a.best_history == b.best_history


def test_elitism_best_fitness_never_decreases():
    res = run_ga(lambda x: np.cos(x) - x, GAConfig(seed=7, tolerance=0))
    assert all(b2 >= b1 for b1, b2 in zip(res.best_history, res.best_history[1:]))


def test_odd_population_size():
    res = run_ga(lambda x: x + 1, GAConfig(pop_size=51, seed=0))
    assert len(res.final_population) == 51


def test_early_stopping():
    res = run_ga(lambda x: x - 3, GAConfig(generations=500, tolerance=1e-3, seed=1))
    assert res.n_generations < 500


def test_invalid_config():
    with pytest.raises(ValueError):
        GAConfig(lower_bound=5, upper_bound=1)


def test_patience_stops_when_stuck():
    res = run_ga(lambda x: x**2 + 1, GAConfig(generations=500, patience=20, seed=0))
    assert res.n_generations < 500


def test_patience_none_runs_all_generations():
    res = run_ga(lambda x: x**2 + 1, GAConfig(generations=50, patience=None, seed=0))
    assert res.n_generations == 50