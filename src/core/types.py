from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
import numpy as np

_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

try:
    from ga_solver.config import GAConfig
    from ga_solver.parser.expression import ParseError, Problem
except ImportError:
    from src.ga_solver.config import GAConfig
    from src.ga_solver.parser.expression import ParseError, Problem


@dataclass
class SolveRequest:
    """Yêu cầu giải phương trình từ Frontend."""
    equations: list[str]
    bounds: dict[str, tuple[float, float]]
    config: GAConfig
    run_baseline: bool = False


@dataclass
class GARunResult:
    """Kết quả chạy thuật toán di truyền GA."""
    roots: list[np.ndarray | list[float]] = field(default_factory=list)
    n_evaluations: int = 0
    elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)
    history_best: list[float] = field(default_factory=list)
    history_mean: list[float] = field(default_factory=list)


@dataclass
class BaselineResult:
    """Kết quả chạy phương pháp cổ điển (Baseline)."""
    method: str
    roots: list[np.ndarray | list[float]] = field(default_factory=list)
    elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class SolveResponse:
    """Phản hồi kết quả giải phương trình cho giao diện Streamlit."""
    ok: bool
    problem: Problem | None = None
    ga: GARunResult | None = None
    baseline: BaselineResult | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
