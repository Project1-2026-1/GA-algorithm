"""Chuyển chuỗi người dùng nhập thành hàm f(x) (nhận số hoặc mảng numpy).

Cách dùng:
    from ga_solver.parser.expression import parse_function
    f = parse_function("2x^2 + sin(x)")
    f(3)                        # -> 18.141120008059868 (float)
    f(np.linspace(-1, 1, 5))    # -> mảng numpy cùng kích thước (vectorized)

Quy ước:
  * x là số -> trả về float; x là mảng numpy -> trả về mảng float cùng shape (dùng được với GA/solver).
  * Nếu biểu thức không xác định tại x (chia 0, ln(-1), sqrt(-4), tràn số, kết quả phức)
    thì giá trị đó là nan (kiểm tra bằng math.isnan / np.isnan), không ném lỗi. inf cũng được đổi thành nan.
  * Chuỗi nhập sai (cú pháp, tên lạ, ký tự lạ) thì parse_function ném ValueError với thông báo tiếng Việt.
  * Nếu chuỗi có dấu '=' thì hiểu là phương trình và f(x) = VT - VP.
  * Không dùng eval trên chuỗi thô: chuỗi được tách token, kiểm tra danh sách trắng rồi mới biên dịch.

Cú pháp hỗ trợ:
  * Toán tử: + - * / % ^ (hoặc **), ngoặc ( ), dấu trừ đơn ngôi, ^ kết hợp phải (2^3^2 = 2^9)
  * Nhân ngầm: 2x, 3x^2, 2sin(x), x(x+1), (x+1)(x-1), 2pi, xsin(x)
  * Hàm: sin cos tan asin acos atan sinh cosh tanh exp ln log log10 log2 sqrt abs  (log = ln)
  * Hằng số: pi, e. Biến: x. Không phân biệt hoa/thường.
  * Số: 3, 2.5, 2,5 (dấu phẩy thập phân), .5, 1e-3, 2E5
"""
import ast
import math
import re

import numpy as np

__all__ = ["parse_function"]

MAX_LEN = 1000

FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "ln": np.log,
    "log10": np.log10, "log2": np.log2,
    "sqrt": np.sqrt, "abs": np.abs,
}
CONSTS = {"pi": math.pi, "e": math.e}
_NAMES = sorted([*FUNCS, *CONSTS, "x"], key=len, reverse=True)   # khớp tên dài trước (log10 trước log)

_TOKEN = re.compile(
    r"(?P<num>(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:e[+-]?[0-9]+)?)"
    r"|(?P<word>[a-z]+[0-9]*)"
    r"|(?P<op>\*\*|[-+*/%^()])"
)

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd,
)


def _split_word(w):
    """Tách 'xsin' -> [x, sin], 'x2' -> [x, 2], 'log10' -> [log10]. Trả về None nếu có phần không nhận ra."""
    if not w:
        return []
    if w.isdigit():
        return [("num", w)]
    for name in _NAMES:
        if w.startswith(name):
            rest = _split_word(w[len(name):])
            if rest is not None:
                return [("func" if name in FUNCS else "name", name)] + rest
    return None


def _tokenize(s):
    tokens, pos = [], 0
    while pos < len(s):
        if s[pos].isspace():
            pos += 1
            continue
        m = _TOKEN.match(s, pos)
        if not m:
            raise ValueError(f"Ký tự không hợp lệ: '{s[pos]}'")
        pos = m.end()
        if m.lastgroup == "word":
            parts = _split_word(m.group())
            if parts is None:
                raise ValueError(f"Tên không hợp lệ: '{m.group()}' "
                                 f"(chỉ dùng x, pi, e và các hàm: {', '.join(sorted(FUNCS))})")
            tokens += parts
        else:
            tokens.append((m.lastgroup, m.group()))
    return tokens


def _to_python(side):
    """Chuỗi một vế -> chuỗi biểu thức Python (đã thêm dấu nhân ngầm, ^ -> **)."""
    tokens = _tokenize(side)
    if not tokens:
        raise ValueError("Biểu thức trống")
    out, prev = [], None
    for i, (kind, val) in enumerate(tokens):
        if kind == "func" and not (i + 1 < len(tokens) and tokens[i + 1][1] == "("):
            raise ValueError(f"Hàm '{val}' phải có ngoặc, ví dụ {val}(x)")
        starts_value = kind in ("num", "name", "func") or val == "("
        ends_value = prev is not None and (prev[0] in ("num", "name") or prev[1] == ")")
        if ends_value and starts_value:
            out.append("*")                          # nhân ngầm
        out.append("**" if val == "^" else val)
        prev = (kind, val)
    return "".join(out)


class _ToFloat(ast.NodeTransformer):
    """Đổi mọi hằng số thành float để 9^9^9 không tạo số nguyên khổng lồ làm treo máy."""
    def visit_Constant(self, node):
        return ast.copy_location(ast.Constant(float(node.value)), node)


def _compile(expr):
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, RecursionError, MemoryError):
        raise ValueError("Sai cú pháp biểu thức (kiểm tra ngoặc và toán tử)")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Cú pháp không được phép: {type(node).__name__}")
        if isinstance(node, ast.Call) and not (
                isinstance(node.func, ast.Name) and node.func.id in FUNCS and not node.keywords):
            raise ValueError("Hàm không được hỗ trợ")
        if isinstance(node, ast.Name) and node.id not in FUNCS and node.id not in CONSTS and node.id != "x":
            raise ValueError(f"Tên không hợp lệ: '{node.id}'")
    return compile(ast.fix_missing_locations(_ToFloat().visit(tree)), "<f(x)>", "eval")


def parse_function(text):
    """Chuỗi -> hàm f(x) trả về float (nan nếu không xác định tại x). Ném ValueError nếu chuỗi sai."""
    if not isinstance(text, str):
        raise ValueError("Đầu vào phải là chuỗi")
    if len(text) > MAX_LEN:
        raise ValueError(f"Biểu thức quá dài (tối đa {MAX_LEN} ký tự)")
    s = (text.lower().replace(",", ".").replace("×", "*").replace("÷", "/").replace("−", "-"))
    if s.count("=") > 1:
        raise ValueError("Chỉ được có tối đa một dấu '='")
    if "=" in s:
        left, right = s.split("=")
        expr = f"({_to_python(left)})-({_to_python(right)})"
    else:
        expr = _to_python(s)

    code = _compile(expr)
    env = {"__builtins__": {}, **FUNCS, **CONSTS}

    def f(x):
        try:
            arr = np.asarray(x, dtype=float)
        except (TypeError, ValueError):
            return math.nan
        scalar = arr.ndim == 0
        try:
            with np.errstate(all="ignore"):               # chia 0, log(-1)... -> nan/inf, không ném lỗi
                v = eval(code, env, {"x": arr[()] if scalar else arr})
            if isinstance(v, complex):
                raise ValueError
            out = np.broadcast_to(np.asarray(v, dtype=float), arr.shape).copy()
        except (ValueError, ZeroDivisionError, OverflowError, TypeError, RecursionError):
            out = np.full(arr.shape, np.nan)
        out[~np.isfinite(out)] = np.nan                   # inf cũng coi là không xác định
        return float(out) if scalar else out

    return f


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print('Dùng: python -m ga_solver.parser.expression "<biểu thức>" <x>')
    else:
        print(parse_function(sys.argv[1])(float(sys.argv[2])))
