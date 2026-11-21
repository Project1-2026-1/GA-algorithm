import time
from dataclasses import dataclass, field, replace

import numpy as np

from ga_solver.config import GAConfig, SolverConfig
from ga_solver.ga.algorithm import GAResult, run_ga
from ga_solver.ga.fitness import Func


@dataclass
class Root:
    x: float
    error: float
    generations: int


@dataclass
class SolveResult:
    roots: list[Root] = field(default_factory=list)
    runs: list[GAResult] = field(default_factory=list)
    intervals: list[tuple[float, float]] = field(default_factory=list)
    elapsed_time: float = 0.0

    @property
    def root_values(self) -> list[float]:
        return [r.x for r in self.roots]

    @property
    def found(self) -> bool:
        return len(self.roots) > 0


def find_roots(f: Func, ga_config: GAConfig | None = None,
               solver_config: SolverConfig | None = None) -> SolveResult:
    ga_config = ga_config or GAConfig()
    sc = solver_config or SolverConfig()
    start = time.perf_counter()

    edges = np.linspace(ga_config.lower_bound, ga_config.upper_bound, sc.n_subintervals + 1)
    pending = [(float(lo), float(hi)) for lo, hi in zip(edges[:-1], edges[1:])][::-1]

    roots: list[Root] = []
    runs: list[GAResult] = []
    intervals: list[tuple[float, float]] = []

    while pending and len(runs) < sc.max_ga_runs:
        lo, hi = pending.pop()
        if hi - lo < sc.min_width:
            continue

        attempts = 1 + (sc.max_retries if has_sign_change(f, lo, hi, sc.sign_check_points) else 0)
        root = None
        for _ in range(attempts):
            seed = None if ga_config.seed is None else ga_config.seed + len(runs)
            res = run_ga(f, replace(ga_config, lower_bound=lo, upper_bound=hi, seed=seed))
            runs.append(res)
            intervals.append((lo, hi))
            if res.best_error < sc.root_tolerance:
                root = Root(res.best_x, res.best_error, res.n_generations)
                break

        if root is None:
            continue

        roots.append(root)
        pending.append((root.x + sc.exclusion_radius, hi))
        pending.append((lo, root.x - sc.exclusion_radius))

    return SolveResult(
        roots=merge_roots(f, roots, sc.exclusion_radius, sc.root_tolerance),
        runs=runs,
        intervals=intervals,
        elapsed_time=time.perf_counter() - start,
    )


def has_sign_change(f: Func, lo: float, hi: float, n_points: int) -> bool:
    x = np.linspace(lo, hi, n_points)
    with np.errstate(all="ignore"):
        try:
            y = np.broadcast_to(np.asarray(f(x), dtype=float), x.shape)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            return False
    s = np.sign(y[np.isfinite(y)])
    if np.any(s == 0):
        return True
    return bool(np.any(s[1:] != s[:-1]))


def is_same_root(f: Func, a: float, b: float, root_tolerance: float, n_points: int = 20) -> bool:
    x = np.linspace(a, b, n_points)
    with np.errstate(all="ignore"):
        try:
            y = np.abs(np.broadcast_to(np.asarray(f(x), dtype=float), x.shape))
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            return False
    return bool(np.all(np.isfinite(y)) and np.all(y < root_tolerance))


def merge_roots(f: Func, roots: list[Root], min_gap: float, root_tolerance: float) -> list[Root]:
    groups: list[list[Root]] = []
    for r in sorted(roots, key=lambda r: r.x):
        last = groups[-1][-1] if groups else None
        if last and (r.x - last.x < min_gap or is_same_root(f, last.x, r.x, root_tolerance)):
            groups[-1].append(r)
        else:
            groups.append([r])
    return [min(g, key=lambda r: r.error) for g in groups]