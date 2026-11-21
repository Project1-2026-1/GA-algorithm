from __future__ import annotations

import time
from dataclasses import dataclass, field
import numpy as np
from scipy import optimize

from ga_solver.parser.expression import Problem


@dataclass
class BaselineResult:
    method: str
    roots: list[np.ndarray | list[float]] = field(default_factory=list)
    elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)


def solve_baseline(
    problem: Problem,
    bounds: dict[str, tuple[float, float]],
    n_subintervals: int = 20,
    tolerance: float = 1e-6,
) -> BaselineResult:
    """Giải phương trình / hệ phương trình bằng phương pháp giải tích số cổ điển (SciPy)."""
    start = time.perf_counter()
    n_vars = len(problem.variables)
    B = np.array([bounds.get(v, (-10.0, 10.0)) for v in problem.variables], dtype=float)

    roots: list[np.ndarray] = []

    if n_vars == 1 and problem.n_eqs == 1:
        method_name = "SciPy Brent's Method (Brentq)"
        lo, hi = B[0, 0], B[0, 1]
        grid = np.linspace(lo, hi, n_subintervals + 1)

        def f_scalar(val: float) -> float:
            try:
                res = problem.residuals(np.array([[val]]))[0, 0]
                return float(res)
            except Exception:
                return np.nan

        for i in range(len(grid) - 1):
            a, b = float(grid[i]), float(grid[i + 1])
            fa, fb = f_scalar(a), f_scalar(b)
            if np.isnan(fa) or np.isnan(fb):
                continue
            if abs(fa) < tolerance:
                roots.append(np.array([a]))
            if fa * fb < 0:
                try:
                    sol = optimize.root_scalar(f_scalar, bracket=[a, b], method="brentq", xtol=tolerance)
                    if sol.converged and np.isfinite(sol.root):
                        roots.append(np.array([float(sol.root)]))
                except Exception:
                    pass

    else:
        method_name = "SciPy Hybrid Powell / Least-Squares"
        rng = np.random.default_rng(42)

        def f_vec(x: np.ndarray) -> np.ndarray:
            return problem.residuals(x.reshape(1, -1))[0]

        # Multi-start từ 20 điểm trong miền tìm kiếm
        for _ in range(25):
            x0 = rng.uniform(B[:, 0], B[:, 1])
            try:
                if problem.n_eqs == n_vars:
                    sol = optimize.root(f_vec, x0, method="hybr", tol=tolerance)
                    x_candidate = sol.x
                else:
                    sol = optimize.least_squares(
                        f_vec, x0, bounds=(B[:, 0], B[:, 1]), ftol=tolerance, xtol=tolerance
                    )
                    x_candidate = sol.x

                if np.all(np.isfinite(x_candidate)):
                    if np.all(x_candidate >= B[:, 0] - 1e-4) and np.all(x_candidate <= B[:, 1] + 1e-4):
                        if problem.max_abs_residual(x_candidate) < tolerance * 10:
                            roots.append(np.array(x_candidate))
            except Exception:
                pass

    # Loại bỏ nghiệm trùng
    unique_roots: list[np.ndarray] = []
    for r in roots:
        if not any(np.linalg.norm(r - u) < 1e-4 for u in unique_roots):
            unique_roots.append(r)

    # Sắp xếp nghiệm
    if n_vars == 1:
        unique_roots.sort(key=lambda r: float(r[0]))
    else:
        unique_roots.sort(key=lambda r: float(r[0]))

    elapsed = time.perf_counter() - start
    return BaselineResult(
        method=method_name,
        roots=unique_roots,
        elapsed_s=elapsed,
        notes=[f"Đã tìm thấy {len(unique_roots)} nghiệm bằng {method_name}"],
    )
