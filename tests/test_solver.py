import numpy as np
import pytest

from ga_solver.config import GAConfig, SolverConfig
from ga_solver.solver import Root, find_roots, has_sign_change, is_same_root, merge_roots


def roots_of(f, lo, hi, **solver_kwargs):
    res = find_roots(f, GAConfig(lower_bound=lo, upper_bound=hi, seed=0),
                     SolverConfig(**solver_kwargs))
    return res.root_values


def test_quadratic_two_roots():
    assert roots_of(lambda x: x**2 - 4, -10, 10) == pytest.approx([-2, 2], abs=1e-4)


def test_sin_many_roots():
    expected = [k * np.pi for k in range(-3, 4)]
    assert roots_of(np.sin, -10, 10) == pytest.approx(expected, abs=1e-4)


def test_user_exponential_equation():
    f = lambda x: np.exp(5 * x) - 7 * x * 10 + 6 * x * 4 - 102
    assert roots_of(f, -10, 10) == pytest.approx([-2.2173910, 0.9994055], abs=1e-5)


def test_cubic_three_roots():
    f = lambda x: (x + 3) * (x - 1) * (x - 4)
    assert roots_of(f, -10, 10) == pytest.approx([-3, 1, 4], abs=1e-4)


def test_no_real_root():
    assert roots_of(lambda x: x**2 + 1, -10, 10) == []


def test_domain_errors_are_handled():
    assert roots_of(np.log, -5, 5) == pytest.approx([1], abs=1e-4)


def test_double_root():
    assert roots_of(lambda x: (x - 1) ** 2, -5, 5) == pytest.approx([1], abs=1e-3)


def test_root_on_subinterval_boundary_not_duplicated():
    assert roots_of(lambda x: x, -10, 10) == pytest.approx([0], abs=1e-4)


def test_result_metadata():
    res = find_roots(lambda x: x - 1, GAConfig(seed=0), SolverConfig(n_subintervals=5))
    assert len(res.runs) == len(res.intervals) >= 5
    assert res.elapsed_time > 0
    assert res.found


def test_two_roots_in_same_subinterval():
    f = lambda x: (x - 0.2) * (x - 0.7)
    for seed in range(5):
        res = find_roots(f, GAConfig(seed=seed))
        assert res.root_values == pytest.approx([0.2, 0.7], abs=1e-4)


def test_close_roots_inside_one_subinterval():
    f = lambda x: (x - 0.31) * (x - 0.32)
    assert roots_of(f, -10, 10) == pytest.approx([0.31, 0.32], abs=1e-4)


def test_single_subinterval_finds_all_roots():
    f = lambda x: (x + 3) * (x - 1) * (x - 4)
    assert roots_of(f, -10, 10, n_subintervals=1) == pytest.approx([-3, 1, 4], abs=1e-4)


def test_dense_roots_sin_10x():
    expected = [k * np.pi / 10 for k in range(-31, 32)]
    assert roots_of(lambda x: np.sin(10 * x), -10, 10) == pytest.approx(expected, abs=1e-4)


def test_max_ga_runs_limit():
    res = find_roots(lambda x: np.sin(10 * x), GAConfig(seed=0), SolverConfig(max_ga_runs=10))
    assert len(res.runs) <= 10 + SolverConfig().max_retries


def test_has_sign_change():
    assert has_sign_change(lambda x: x - 0.5, 0, 1, 50)
    assert has_sign_change(np.sin, -1, 0, 50)
    assert not has_sign_change(lambda x: x**2 + 1, -1, 1, 50)
    assert not has_sign_change(np.log, -2, -1, 50)


def test_merge_roots_keeps_smallest_error():
    f = lambda x: (x - 1) * (x - 3)
    roots = [Root(1.0, 1e-7, 10), Root(1.00001, 1e-9, 10), Root(3.0, 1e-8, 10)]
    merged = merge_roots(f, roots, min_gap=1e-4, root_tolerance=1e-6)
    assert [r.x for r in merged] == [1.00001, 3.0]


def test_merge_roots_flat_double_root():
    f = lambda x: (x - 1) ** 2
    roots = [Root(1.0, 0.0, 10), Root(1.0004, 1.6e-7, 10), Root(1.0008, 6.4e-7, 10)]
    assert [r.x for r in merge_roots(f, roots, 1e-4, 1e-6)] == [1.0]


def test_is_same_root():
    assert is_same_root(lambda x: (x - 1) ** 2, 1.0, 1.0005, 1e-6)
    assert not is_same_root(lambda x: (x - 0.31) * (x - 0.32), 0.31, 0.32, 1e-6)