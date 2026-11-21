from __future__ import annotations

import re
from typing import Callable
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

try:
    from sympy.parsing.latex import parse_latex
    HAS_PARSE_LATEX = True
except Exception:
    HAS_PARSE_LATEX = False


class ParseError(Exception):
    """Lỗi phân tích cú pháp biểu thức hoặc phương trình."""
    pass


_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

_CONSTANT_NAMES = {"e", "pi", "E", "I", "pi", "exp"}


def clean_latex(s: str) -> str:
    """Chuyển đổi chuỗi LaTeX cơ bản sang biểu thức văn bản."""
    s = s.strip()
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\cdot", "*").replace(r"\times", "*")
    s = s.replace(r"\le", "<=").replace(r"\ge", ">=")
    s = s.replace(r"\neq", "!=")
    
    # \frac{a}{b} -> ((a)/(b))
    for _ in range(5):
        s_new = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
        if s_new == s:
            break
        s = s_new
        
    # \sqrt{a} -> sqrt(a)
    for _ in range(5):
        s_new = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
        if s_new == s:
            break
        s = s_new

    s = re.sub(r"\\sin", "sin", s)
    s = re.sub(r"\\cos", "cos", s)
    s = re.sub(r"\\tan", "tan", s)
    s = re.sub(r"\\ln", "log", s)
    s = re.sub(r"\\log", "log", s)
    s = re.sub(r"\\exp", "exp", s)
    s = re.sub(r"\\pi", "pi", s)
    s = re.sub(r"\\mathrm\{e\}", "E", s)
    s = re.sub(r"\\mathrm\{pi\}", "pi", s)
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace(r"\ ", " ")
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)
    return s


def parse_single_equation(eq_str: str) -> sp.Expr:
    """Phân tích một phương trình (dạng chuỗi hoặc LaTeX) thành sympy.Expr biểu diễn f(X) = 0."""
    eq_str = eq_str.strip()
    if not eq_str:
        raise ParseError("Chuỗi phương trình trống.")

    if "=" in eq_str:
        parts = eq_str.split("=", 1)
        lhs_str, rhs_str = parts[0].strip(), parts[1].strip()
    else:
        lhs_str, rhs_str = eq_str, "0"

    expr: sp.Expr | None = None

    # Thử parse_latex trước nếu có ký hiệu LaTeX
    if HAS_PARSE_LATEX and ("\\" in lhs_str or "\\" in rhs_str):
        try:
            lhs_sym = parse_latex(lhs_str) if lhs_str else sp.Integer(0)
            rhs_sym = parse_latex(rhs_str) if rhs_str else sp.Integer(0)
            expr = sp.simplify(lhs_sym - rhs_sym)
        except Exception:
            expr = None

    if expr is None:
        try:
            clean_l = clean_latex(lhs_str)
            clean_r = clean_latex(rhs_str)
            lhs_sym = parse_expr(clean_l, transformations=_TRANSFORMS) if clean_l else sp.Integer(0)
            rhs_sym = parse_expr(clean_r, transformations=_TRANSFORMS) if clean_r else sp.Integer(0)
            expr = sp.simplify(lhs_sym - rhs_sym)
        except Exception as e:
            raise ParseError(f"Không thể phân tích phương trình '{eq_str}': {e}") from e

    # Chuẩn hoá các hằng số e, pi
    subs_dict = {}
    for sym in expr.free_symbols:
        if sym.name == "e":
            subs_dict[sym] = sp.E
        elif sym.name == "pi":
            subs_dict[sym] = sp.pi
    if subs_dict:
        expr = expr.subs(subs_dict)

    return expr


class Problem:
    """Đại diện cho bài toán giải phương trình hoặc hệ phương trình số thực."""

    def __init__(self, raw_equations: list[str], exprs: list[sp.Expr], variables: list[str]):
        self.raw_equations = list(raw_equations)
        self.exprs = list(exprs)
        self.variables = list(variables)
        self.n_eqs = len(exprs)
        self.equations = [str(e) for e in exprs]

        # Chuẩn bị lambdify cho NumPy vectorised evaluation
        sym_vars = [sp.Symbol(v) for v in self.variables]
        self._lambdas: list[Callable] = []
        for e in self.exprs:
            try:
                fn = sp.lambdify(sym_vars, e, modules=["numpy", {"log": np.log, "ln": np.log}])
            except Exception:
                fn = sp.lambdify(sym_vars, e, modules=["numpy"])
            self._lambdas.append(fn)

    def residuals(self, X: np.ndarray) -> np.ndarray:
        """
        Tính phần dư (residual) của từng phương trình.
        X: shape (N, n_vars) hoặc (n_vars,)
        Trả về: shape (N, n_eqs)
        """
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        N, D = arr.shape
        if D != len(self.variables):
            raise ValueError(f"Kích thước X ({D}) không khớp số lượng biến ({len(self.variables)})")

        args = [arr[:, i] for i in range(D)]
        res_list = []
        with np.errstate(all="ignore"):
            for fn in self._lambdas:
                try:
                    val = fn(*args)
                    val = np.asarray(val, dtype=float)
                    if val.shape != (N,):
                        val = np.broadcast_to(val, (N,)).copy()
                except Exception:
                    # Fallback điểm từng điểm
                    row_vals = []
                    for k in range(N):
                        pt_args = [arr[k, i] for i in range(D)]
                        try:
                            v = float(fn(*pt_args))
                        except Exception:
                            v = np.inf
                        row_vals.append(v)
                    val = np.array(row_vals, dtype=float)

                # Thay thế nan hoặc không xác định bằng inf
                val = np.nan_to_num(val, nan=np.inf, posinf=np.inf, neginf=-np.inf)
                res_list.append(val)

        return np.column_stack(res_list) if res_list else np.zeros((N, 0))

    def objective(self, X: np.ndarray) -> np.ndarray:
        """
        Mục tiêu tối ưu hoá = tổng bình phương các phần dư.
        Trả về shape (N,)
        """
        res = self.residuals(X)
        return np.sum(res**2, axis=1)

    def max_abs_residual(self, root: np.ndarray | list[float]) -> float:
        """Tính sai số phần dư lớn nhất tại nghiệm root."""
        arr = np.asarray(root, dtype=float).reshape(1, -1)
        res = self.residuals(arr)[0]
        return float(np.max(np.abs(res)))


def detect_variables(equations: list[str]) -> tuple[list[str], list[str]]:
    """Phát hiện các biến số từ danh sách phương trình và đưa ra cảnh báo nếu có."""
    clean_eqs = [e.strip() for e in equations if e.strip()]
    if not clean_eqs:
        return [], ["Vui lòng nhập ít nhất một phương trình."]

    all_symbols: set[str] = set()
    warnings: list[str] = []

    for eq in clean_eqs:
        expr = parse_single_equation(eq)
        for s in expr.free_symbols:
            if s.name not in _CONSTANT_NAMES:
                all_symbols.add(s.name)

    variables = sorted(list(all_symbols))
    if not variables:
        warnings.append("Không phát hiện biến số nào trong biểu thức.")
    return variables, warnings


def parse_problem(equations: list[str]) -> Problem:
    """Phân tích danh sách phương trình thành đối tượng Problem."""
    clean_eqs = [e.strip() for e in equations if e.strip()]
    if not clean_eqs:
        raise ParseError("Danh sách phương trình trống.")

    exprs: list[sp.Expr] = []
    all_symbols: set[str] = set()

    for eq in clean_eqs:
        expr = parse_single_equation(eq)
        exprs.append(expr)
        for s in expr.free_symbols:
            if s.name not in _CONSTANT_NAMES:
                all_symbols.add(s.name)

    variables = sorted(list(all_symbols))
    if not variables:
        variables = ["x"]

    return Problem(raw_equations=clean_eqs, exprs=exprs, variables=variables)
