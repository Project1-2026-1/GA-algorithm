import math
import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from ga_solver.parser.expression import parse_function


def ev(text, x=0.0):
    return parse_function(text)(x)


class Base(unittest.TestCase):
    def assertNan(self, v):
        self.assertTrue(v != v, f"mong đợi nan, nhận {v!r}")


class TestArithmetic(Base):
    def test_constant_and_variable(self):
        self.assertEqual(ev("5"), 5.0)
        self.assertEqual(ev("x", 3), 3.0)
        self.assertEqual(ev("2+3*4"), 14.0)

    def test_power_and_precedence(self):
        self.assertAlmostEqual(ev("x^2", 3), 9)
        self.assertAlmostEqual(ev("x**2", 3), 9)
        self.assertAlmostEqual(ev("-x^2", 3), -9)
        self.assertAlmostEqual(ev("-2^2"), -4)
        self.assertAlmostEqual(ev("2^3^2"), 512)          # kết hợp phải
        self.assertAlmostEqual(ev("x^-1", 2), 0.5)
        self.assertAlmostEqual(ev("8/4/2"), 1)
        self.assertAlmostEqual(ev("7%4"), 3)
        self.assertAlmostEqual(ev("2*(x+1)", 2), 6)

    def test_unary(self):
        self.assertAlmostEqual(ev("--x", 3), 3)
        self.assertAlmostEqual(ev("+x", 3), 3)
        self.assertAlmostEqual(ev("3-(-x)", 2), 5)

    def test_return_is_float(self):
        for text in ("5", "x", "x^2", "x/2"):
            self.assertIsInstance(ev(text, 2), float)

    def test_int_and_float_input(self):
        self.assertAlmostEqual(ev("x^2", 3), ev("x^2", 3.0))


class TestImplicitMultiplication(Base):
    def test_number_variable(self):
        self.assertAlmostEqual(ev("2x", 4), 8)
        self.assertAlmostEqual(ev("3x^2", 2), 12)
        self.assertAlmostEqual(ev("x2", 4), 8)
        self.assertAlmostEqual(ev("2.5x", 2), 5)

    def test_with_functions_and_constants(self):
        self.assertAlmostEqual(ev("2sin(x)", math.pi / 2), 2)
        self.assertAlmostEqual(ev("2pi"), 2 * math.pi)
        self.assertAlmostEqual(ev("xsin(x)", math.pi / 2), math.pi / 2)
        self.assertAlmostEqual(ev("xe^x", 1), math.e ** 1 * 1)
        self.assertAlmostEqual(ev("2e^x", 0), 2)           # 2 * e^x, không phải 2e^...

    def test_parentheses(self):
        self.assertAlmostEqual(ev("(x+1)(x-1)", 3), 8)
        self.assertAlmostEqual(ev("x(x+1)", 2), 6)
        self.assertAlmostEqual(ev("2(x+1)", 2), 6)
        self.assertAlmostEqual(ev("(x+1)x", 2), 6)
        self.assertAlmostEqual(ev("(x)2", 3), 6)


class TestNumbers(Base):
    def test_scientific_notation(self):
        self.assertAlmostEqual(ev("1e3*x", 2), 2000)
        self.assertAlmostEqual(ev("x*1e-3", 2), 0.002)
        self.assertAlmostEqual(ev("2E2"), 200)

    def test_decimal_forms(self):
        self.assertAlmostEqual(ev(".5x", 4), 2)
        self.assertAlmostEqual(ev("2,5x", 2), 5)            # dấu phẩy thập phân kiểu Việt Nam
        self.assertAlmostEqual(ev("3."), 3)

    def test_huge_literal(self):
        self.assertEqual(ev("99999999999999999999999999999"), 1e29)


class TestFunctions(Base):
    def test_trig(self):
        self.assertAlmostEqual(ev("sin(x)", math.pi / 6), 0.5)
        self.assertAlmostEqual(ev("cos(0)"), 1)
        self.assertAlmostEqual(ev("tan(x)", math.pi / 4), 1)
        self.assertAlmostEqual(ev("asin(1)"), math.pi / 2)
        self.assertAlmostEqual(ev("acos(1)"), 0)
        self.assertAlmostEqual(ev("atan(1)"), math.pi / 4)

    def test_hyperbolic(self):
        self.assertAlmostEqual(ev("sinh(x)", 1), math.sinh(1))
        self.assertAlmostEqual(ev("cosh(x)", 1), math.cosh(1))
        self.assertAlmostEqual(ev("tanh(x)", 1), math.tanh(1))

    def test_exp_log(self):
        self.assertAlmostEqual(ev("exp(x)", 1), math.e)
        self.assertAlmostEqual(ev("e^x", 2), math.e ** 2)
        self.assertAlmostEqual(ev("ln(e)"), 1)
        self.assertAlmostEqual(ev("log(e^2)"), 2)           # log = ln
        self.assertAlmostEqual(ev("log10(1000)"), 3)
        self.assertAlmostEqual(ev("log2(8)"), 3)
        self.assertAlmostEqual(ev("log10(x)", 100), 2)

    def test_sqrt_abs(self):
        self.assertAlmostEqual(ev("sqrt(16)"), 4)
        self.assertAlmostEqual(ev("abs(x)", -3), 3)
        self.assertAlmostEqual(ev("abs(-3x)", 2), 6)

    def test_nested(self):
        self.assertAlmostEqual(ev("sin(cos(x))", 0), math.sin(1))
        self.assertAlmostEqual(ev("sqrt(abs(x)+1)", -3), 2)


class TestConstantsAndCase(Base):
    def test_constants(self):
        self.assertAlmostEqual(ev("pi"), math.pi)
        self.assertAlmostEqual(ev("e"), math.e)

    def test_case_and_spaces(self):
        self.assertAlmostEqual(ev("  2 * x  ", 3), 6)
        self.assertAlmostEqual(ev("SIN(X)", math.pi / 2), 1)
        self.assertAlmostEqual(ev("PI"), math.pi)

    def test_unicode_operators(self):
        self.assertAlmostEqual(ev("2×x", 3), 6)
        self.assertAlmostEqual(ev("x÷2", 3), 1.5)
        self.assertAlmostEqual(ev("x−1", 3), 2)


class TestEquationForm(Base):
    def test_equation_is_left_minus_right(self):
        self.assertAlmostEqual(ev("x^2 = 4", 3), 5)
        self.assertAlmostEqual(ev("sin(x) = x/2", 0), 0)
        self.assertAlmostEqual(ev("x^3 - 2x - 5 = 0", 2), 8 - 4 - 5)
        self.assertAlmostEqual(ev("2 = x", 5), -3)

    def test_root_gives_zero(self):
        self.assertAlmostEqual(ev("x^2 = 4", 2), 0)
        self.assertAlmostEqual(ev("cos(x) = x", 0.7390851332151607), 0, places=12)


class TestUndefinedIsNan(Base):
    def test_domain_errors(self):
        self.assertNan(ev("sqrt(x)", -1))
        self.assertNan(ev("ln(x)", 0))
        self.assertNan(ev("ln(x)", -1))
        self.assertNan(ev("log10(x)", 0))
        self.assertNan(ev("asin(x)", 2))
        self.assertNan(ev("acos(x)", -2))

    def test_division_by_zero(self):
        self.assertNan(ev("1/x", 0))
        self.assertNan(ev("1/(x-1)", 1))
        self.assertNan(ev("5%x", 0))

    def test_complex_results(self):
        self.assertNan(ev("(-8)^(1/3)"))
        self.assertNan(ev("x^0.5", -4))
        self.assertNan(ev("x^x", -0.5))

    def test_overflow(self):
        self.assertNan(ev("exp(x)", 1000))
        self.assertNan(ev("10^x", 400))
        self.assertNan(ev("sinh(x)", 1000))

    def test_defined_elsewhere(self):
        self.assertAlmostEqual(ev("sqrt(x)", 4), 2)
        self.assertAlmostEqual(ev("1/x", 4), 0.25)

    def test_bad_input_value_gives_nan(self):
        self.assertNan(parse_function("x")("abc"))
        self.assertNan(parse_function("x")(None))


class TestVectorized(Base):
    """solver.py gọi f(mảng numpy), nên f phải nhận cả số lẫn mảng."""

    def test_array_in_array_out(self):
        x = np.linspace(-2, 2, 9)
        y = parse_function("x^2 - 4")(x)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(y.shape, x.shape)
        np.testing.assert_allclose(y, x ** 2 - 4)

    def test_matches_scalar_version(self):
        f = parse_function("sin(x) = x/2")
        x = np.linspace(-5, 5, 21)
        np.testing.assert_allclose(f(x), [f(float(t)) for t in x])

    def test_constant_expression_is_broadcast(self):
        y = parse_function("5")(np.linspace(0, 1, 4))
        np.testing.assert_array_equal(y, [5, 5, 5, 5])
        y = parse_function("2^3")(np.zeros(3))
        np.testing.assert_array_equal(y, [8, 8, 8])

    def test_nan_only_where_undefined(self):
        y = parse_function("sqrt(x)")(np.array([-1.0, 0.0, 4.0]))
        self.assertTrue(np.isnan(y[0]))
        self.assertEqual(y[1], 0.0)
        self.assertEqual(y[2], 2.0)
        y = parse_function("1/x")(np.array([-2.0, 0.0, 2.0]))
        np.testing.assert_array_equal(np.isnan(y), [False, True, False])
        y = parse_function("exp(x)")(np.array([0.0, 1000.0]))
        np.testing.assert_array_equal(np.isnan(y), [False, True])

    def test_shapes_and_types(self):
        f = parse_function("2x + 1")
        self.assertEqual(f(np.zeros((3, 4))).shape, (3, 4))
        self.assertEqual(f(np.array([])).shape, (0,))
        np.testing.assert_allclose(f([1, 2, 3]), [3, 5, 7])      # list
        np.testing.assert_allclose(f(np.array([1, 2, 3])), [3, 5, 7])  # mảng int
        self.assertIsInstance(f(np.float64(2.0)), float)

    def test_does_not_modify_input(self):
        x = np.array([1.0, 2.0, 3.0])
        parse_function("x^2")(x)
        np.testing.assert_array_equal(x, [1, 2, 3])

    def test_solver_style_usage(self):
        """Mô phỏng has_sign_change / is_same_root trong solver.py."""
        f = parse_function("x^3 - 2x - 5 = 0")
        x = np.linspace(2.0, 2.2, 20)
        y = np.broadcast_to(np.asarray(f(x), dtype=float), x.shape)
        s = np.sign(y[np.isfinite(y)])
        self.assertTrue(bool(np.any(s[1:] != s[:-1])))             # đổi dấu quanh nghiệm 2.0945


class TestInvalidInput(Base):
    def assertBad(self, text):
        with self.assertRaises(ValueError, msg=repr(text)):
            parse_function(text)

    def test_empty(self):
        for t in ("", "   ", "()"):
            self.assertBad(t)

    def test_syntax(self):
        for t in ("x+", "*x", "(x", "x)", "2**", "x++*2", "((x)", "x +* 2"):
            self.assertBad(t)

    def test_equation_syntax(self):
        for t in ("x = ", "= x", "=", "x = y = z", "x == 1", "x <= 1"):
            self.assertBad(t)

    def test_function_without_parentheses(self):
        for t in ("sin", "sin x", "sin+1", "2cos", "log10x"):
            self.assertBad(t)

    def test_unknown_names(self):
        for t in ("y+1", "foo(x)", "a*x", "z", "xy", "print(x)", "f(x)", "if"):
            self.assertBad(t)

    def test_invalid_characters(self):
        for t in ("x²", "x;1", "x#1", "'a'", "[1,2]", "{x}", "x!", "x&1", "x@2", "x$"):
            self.assertBad(t)

    def test_non_string(self):
        for t in (None, 5, ["x"]):
            self.assertBad(t)


class TestSecurity(Base):
    def test_injection_attempts(self):
        for t in ("__import__('os').system('echo hi')", "open('f')", "x.__class__",
                  "().__class__.__bases__", "exec('1')", "eval('1')", "lambda: 1",
                  "x if 1 else 2", "[i for i in x]", "globals()", "x.real"):
            with self.assertRaises(ValueError, msg=t):
                parse_function(t)

    def test_huge_power_does_not_hang(self):
        t0 = time.time()
        self.assertNan(ev("9^9^9^9"))
        self.assertNan(ev("9^9^9"))
        self.assertLess(time.time() - t0, 2)

    def test_too_long(self):
        with self.assertRaises(ValueError):
            parse_function("x+" * 600 + "x")

    def test_deep_nesting(self):
        with self.assertRaises(ValueError):
            parse_function("(" * 300 + "x" + ")" * 300)


if __name__ == "__main__":
    unittest.main()
