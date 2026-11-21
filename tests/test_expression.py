import numpy as np
import pytest

from ga_solver.parser.expression import (
    ParseError,
    detect_variables,
    parse_problem,
    parse_single_equation,
)
from src.core import api
from src.core.types import GAConfig, SolveRequest


def test_parse_simple_linear():
    expr = parse_single_equation("x + 9 = 5")
    assert str(expr) in ["x + 4", "4 + x"]


def test_parse_quadratic():
    expr = parse_single_equation("x^2 - 4 = 0")
    assert str(expr) in ["x**2 - 4", "-4 + x**2"]


def test_parse_latex_fraction():
    expr = parse_single_equation(r"\frac{x}{2} + 1 = 0")
    assert str(expr) in ["x/2 + 1", "1 + x/2"]


def test_parse_latex_trig():
    expr = parse_single_equation(r"\sin(x) = 0.5")
    assert str(expr) in ["sin(x) - 0.5", "-0.5 + sin(x)"]


def test_detect_variables():
    vars1, warns1 = detect_variables(["x + 9 = 5"])
    assert vars1 == ["x"]
    assert len(warns1) == 0

    vars2, warns2 = detect_variables(["x^2 + y^2 = 4", "x - y = 1"])
    assert vars2 == ["x", "y"]

    vars3, warns3 = detect_variables([r"e^{x} - \pi = 0"])
    assert vars3 == ["x"]


def test_problem_residuals_and_objective():
    prob = parse_problem(["x + 9 = 5"])
    assert prob.variables == ["x"]
    assert prob.n_eqs == 1

    # At x = -4: x + 9 - 5 = 0
    res = prob.residuals(np.array([[-4.0]]))
    assert np.isclose(res[0, 0], 0.0)
    assert np.isclose(prob.objective(np.array([[-4.0]]))[0], 0.0)

    # At x = 0: 0 + 4 = 4
    res0 = prob.residuals(np.array([[0.0]]))
    assert np.isclose(res0[0, 0], 4.0)
    assert np.isclose(prob.objective(np.array([[0.0]]))[0], 16.0)


def test_api_solve_1d():
    req = SolveRequest(
        equations=["x^2 - 4 = 0"],
        bounds={"x": (-10.0, 10.0)},
        config=GAConfig(pop_size=50, generations=100, seed=42),
        run_baseline=True,
    )
    resp = api.solve(req)
    assert resp.ok
    assert resp.ga is not None
    assert len(resp.ga.roots) >= 1
    # Check that roots are near -2 or 2
    root_vals = [float(r[0]) for r in resp.ga.roots]
    assert any(np.isclose(v, 2.0, atol=1e-2) for v in root_vals) or any(np.isclose(v, -2.0, atol=1e-2) for v in root_vals)


def test_api_solve_multivariable():
    req = SolveRequest(
        equations=["x + y = 3", "x - y = 1"],
        bounds={"x": (-10.0, 10.0), "y": (-10.0, 10.0)},
        config=GAConfig(pop_size=50, generations=100, seed=42),
        run_baseline=True,
    )
    resp = api.solve(req)
    assert resp.ok
    assert resp.ga is not None
    assert len(resp.ga.roots) >= 1
    r = resp.ga.roots[0]
    # Exact root: x = 2, y = 1
    assert np.isclose(float(r[0]), 2.0, atol=1e-2)
    assert np.isclose(float(r[1]), 1.0, atol=1e-2)
