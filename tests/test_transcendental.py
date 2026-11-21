import numpy as np
import pytest
from scipy.special import j0

from ga_solver.config import GAConfig
from ga_solver.solver import find_roots

CASES = [
    pytest.param(
        lambda x: x * np.exp(x) - 1, -10, 10,
        [0.5671432904],
        id="x*e^x = 1",
    ),
    pytest.param(
        lambda x: x - np.cos(x), -10, 10,
        [0.7390851332],
        id="x = cos(x)",
    ),
    pytest.param(
        lambda x: np.exp(-x) - np.sin(x), 0, 10,
        [0.5885327440, 3.0963639324, 6.2850492734, 9.4246972547],
        id="e^-x = sin(x)",
    ),
    pytest.param(
        lambda x: np.log(x) + x - 2, -5, 5,
        [1.5571455990],
        id="ln(x) + x = 2",
    ),
    pytest.param(
        lambda x: x**2 - 2**x, -10, 10,
        [-0.7666646960, 2.0, 4.0],
        id="x^2 = 2^x",
    ),
    pytest.param(
        lambda x: x**x - 2, -5, 5,
        [1.5596104695],
        id="x^x = 2",
    ),
    pytest.param(
        lambda x: np.cosh(x) - 3, -10, 10,
        [-1.7627471740, 1.7627471740],
        id="cosh(x) = 3",
    ),
    pytest.param(
        lambda x: np.exp(-x**2) - 0.5, -10, 10,
        [-0.8325546112, 0.8325546112],
        id="e^(-x^2) = 0.5",
    ),
    pytest.param(
        lambda x: np.sin(x) - 0.5 * x, -10, 10,
        [-1.8954942670, 0.0, 1.8954942670],
        id="sin(x) = x/2",
    ),
    pytest.param(
        lambda x: np.exp(np.sin(x)) - 2, -10, 10,
        [-5.5173391124, -3.9074388484, 0.7658461948, 2.3757464588, 7.0490315020, 8.6589317660],
        id="e^sin(x) = 2",
    ),
    pytest.param(
        lambda x: np.tan(x) - x, -10, 10,
        [-7.7252518369, -4.4934094579, 0.0, 4.4934094579, 7.7252518369],
        id="tan(x) = x",
    ),
    pytest.param(
        lambda x: x * np.sin(1 / x), 0.05, 1,
        [1 / (k * np.pi) for k in range(6, 0, -1)],
        id="x*sin(1/x) = 0",
    ),
    pytest.param(
        j0, 0, 20,
        [2.4048255577, 5.5200781103, 8.6537279129, 11.7915344390, 14.9309177085, 18.0710639679],
        id="Bessel J0(x) = 0",
    ),
    pytest.param(
        lambda x: np.sin(x) ** 2, -5, 5,
        [-np.pi, 0.0, np.pi],
        id="sin^2(x) = 0",
    ),
]


@pytest.mark.parametrize("f, lo, hi, expected", CASES)
def test_transcendental_equations(f, lo, hi, expected):
    res = find_roots(f, GAConfig(lower_bound=lo, upper_bound=hi, seed=0))

    assert len(res.roots) == len(expected), (
        f"Tìm được {len(res.roots)} nghiệm {np.round(res.root_values, 6).tolist()}, "
        f"mong đợi {len(expected)}"
    )
    assert res.root_values == pytest.approx(expected, abs=1e-3)
    for r in res.roots:
        assert abs(f(r.x)) < 1e-6


@pytest.mark.parametrize("seed", range(5))
def test_transcendental_stable_across_seeds(seed):
    f = lambda x: np.exp(np.sin(x)) - 2
    expected = [-5.5173391124, -3.9074388484, 0.7658461948, 2.3757464588, 7.0490315020, 8.6589317660]
    res = find_roots(f, GAConfig(seed=seed))
    assert res.root_values == pytest.approx(expected, abs=1e-4)


def test_no_root_transcendental():
    res = find_roots(lambda x: np.exp(x) + np.cos(x) ** 2 + 0.1, GAConfig(seed=0))
    assert not res.found