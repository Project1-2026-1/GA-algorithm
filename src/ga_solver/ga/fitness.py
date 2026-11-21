from typing import Callable

import numpy as np

Func = Callable[[np.ndarray], np.ndarray]


def evaluate_error(f: Func, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    with np.errstate(all="ignore"):
        try:
            y = np.asarray(f(x), dtype=float)
            y = np.broadcast_to(y, x.shape).copy()
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            y = np.array([_safe_scalar(f, xi) for xi in x.ravel()]).reshape(x.shape)
    err = np.abs(y)
    err[~np.isfinite(err)] = np.inf
    return err


def _safe_scalar(f: Func, xi: float) -> float:
    try:
        return float(f(xi))
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return np.inf


def fitness_from_error(err: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + err)