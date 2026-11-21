import matplotlib.pyplot as plt
import numpy as np

from ga_solver.ga.algorithm import GAResult
from ga_solver.ga.fitness import Func


def plot_convergence(result: GAResult):
    fig, ax = plt.subplots()
    ax.plot(result.best_history, label="Best fitness")
    ax.plot(result.mean_history, label="Mean fitness")
    ax.set_xlabel("Thế hệ")
    ax.set_ylabel("Fitness")
    ax.set_title("Quá trình hội tụ")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_function(f: Func, lower: float, upper: float,
                  roots: list[float] | None = None,
                  population: np.ndarray | None = None, n_points: int = 1000):
    x = np.linspace(lower, upper, n_points)
    with np.errstate(all="ignore"):
        y = np.asarray(f(x), dtype=float)
    y[~np.isfinite(y)] = np.nan

    fig, ax = plt.subplots()
    ax.plot(x, y, label="f(x)")
    ax.axhline(0, color="gray", linewidth=0.8)
    if population is not None:
        with np.errstate(all="ignore"):
            ax.scatter(population, f(population), s=10, alpha=0.4, label="Quần thể cuối")
    if roots:
        ax.scatter(roots, np.zeros(len(roots)), color="red", zorder=5, label="Nghiệm")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig