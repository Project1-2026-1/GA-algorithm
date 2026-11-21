import numpy as np

from ga_solver.ga.operators import arithmetic_crossover, gaussian_mutation, tournament_selection


def test_tournament_prefers_better_individuals():
    rng = np.random.default_rng(0)
    pop = np.arange(10, dtype=float)
    fitness = pop.copy()
    selected = tournament_selection(rng, pop, fitness, n_select=1000, k=3)
    assert len(selected) == 1000
    assert selected.mean() > pop.mean()


def test_tournament_k_equal_pop_always_picks_best():
    rng = np.random.default_rng(0)
    pop = np.array([1.0, 2.0, 3.0])
    selected = tournament_selection(rng, pop, pop, n_select=5, k=3)
    assert np.all(selected == 3.0)


def test_crossover_children_between_parents():
    rng = np.random.default_rng(0)
    for _ in range(100):
        c1, c2 = arithmetic_crossover(rng, 2.0, 8.0)
        assert 2.0 <= c1 <= 8.0 and 2.0 <= c2 <= 8.0
        assert np.isclose(c1 + c2, 10.0)


def test_mutation_stays_in_bounds():
    rng = np.random.default_rng(0)
    pop = np.full(1000, 9.9)
    mutated = gaussian_mutation(rng, pop, rate=1.0, std=5.0, lower=-10, upper=10)
    assert mutated.min() >= -10 and mutated.max() <= 10


def test_mutation_rate_zero_keeps_population():
    rng = np.random.default_rng(0)
    pop = np.linspace(-1, 1, 50)
    assert np.array_equal(gaussian_mutation(rng, pop, 0.0, 1.0, -10, 10), pop)


def test_mutation_does_not_modify_input():
    rng = np.random.default_rng(0)
    pop = np.zeros(10)
    gaussian_mutation(rng, pop, 1.0, 1.0, -10, 10)
    assert np.all(pop == 0)