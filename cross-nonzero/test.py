import functools
import itertools
import operator
import unittest

import dual

@functools.lru_cache(maxsize=None)
def stirling(n, k):
  # [[https://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind]]
  if n == 0 and k == 0:
    return 1
  elif n == 0 or k == 0:
    return 0
  else:
    return stirling(n-1, k-1) + stirling(n-1, k) * k

class IterationTest(unittest.TestCase):
  max_ball_count = 12
  max_box_count  = 50

  def test_stirling(self):
    for n in range(self.max_ball_count+1):
      for k in range(self.max_box_count+1):
        d = dual.iter_stirling(range(n), k)
        if (n > 0 and k == 0) or n < k:
          with self.assertRaises(StopIteration):
            next(d)
        else:
          r = set()
          for d in d:
            self.assertEqual(len(d), k)
            self.assertTrue(all(d))
            self.assertCountEqual(itertools.chain.from_iterable(d), range(n))
            r.add(tuple(map(tuple, d)))
          self.assertEqual(len(r), stirling(n, k))

class DualTest:
  @staticmethod
  def format_param(*x):
    return 'parameters are {!r}'.format(x)

class DualExactTest(DualTest):
  def test_add_asso(self):
    for x, y, z in self.sample(3):
      self.assertEqual((x+y)+z, x+(y+z), self.format_param(x, y, z))

  def test_add_comm(self):
    for x, y in self.sample(2, allow_repeats=False):
      self.assertEqual(x+y, y+x, self.format_param(x, y))

  def test_add_iden(self):
    for x, in self.sample():
      self.assertEqual(x+0, x, self.format_param(x))
      self.assertEqual(0+x, x, self.format_param(x))

  def test_add_sub_inv(self):
    for x, y in self.sample(2):
      self.assertEqual(x+y-y, x, self.format_param(x, y))
      self.assertEqual(x-y+y, x, self.format_param(x, y))

  def test_mul_asso(self):
    for x, y, z in self.sample(3):
      self.assertEqual((x*y)*z, x*(y*z), self.format_param(x, y, z))

  def test_mul_comm(self):
    for x, y in self.sample(2, allow_repeats=False):
      self.assertEqual(x*y, y*x, self.format_param(x, y))

  def test_mul_iden(self):
    for x, in self.sample():
      self.assertEqual(x*1, x, self.format_param(x))
      self.assertEqual(1*x, x, self.format_param(x))

  def test_truediv_zero(self):
    for x, y in self.sample(2):
      if y.a == 0:
        with self.assertRaises(ZeroDivisionError):
          x/y

  def test_mul_truediv_inv(self):
    for x, y in self.sample(2):
      if y.a != 0:
        self.assertEqual(x*y/y, x, self.format_param(x, y))
        self.assertEqual(x/y*y, x, self.format_param(x, y))

  def test_add_mul_dist(self):
    for x, y, z in self.sample(3):
      self.assertEqual(x*(y+z), x*y+x*z, self.format_param(x, y, z))
      self.assertEqual((y+z)*x, y*x+z*x, self.format_param(x, y, z))

  def test_pow_zero(self):
    for x, in self.sample():
      self.assertEqual(x**0, 1, self.format_param(x))

  def test_pow_pos(self):
    for x, in self.sample():
      y = 1
      for n in range(1, self.max_pow+1):
        y *= x
        self.assertEqual(x**n, y, self.format_param(x, n))

  def test_pow_neg(self):
    for x, in self.sample():
      if x.a != 0:
        y = 1
        for n in range(1, self.max_pow+1):
          y /= x
          self.assertEqual(x**-n, y, self.format_param(x, n))

  def test_exp_neg(self):
    for x, in self.sample():
      self.assertEqual(dual.exp(-x), 1/dual.exp(x), self.format_param(x))

  def test_log_zero(self):
    for x, in self.sample():
      if x.a == 0:
        with self.assertRaises(ValueError):
          dual.log(x)

  def test_exp_log_inv(self):
    for x, in self.sample():
      self.assert_inv(dual.exp, dual.log, x)

  def test_sin_sym(self):
    for x, in self.sample():
      self.assertEqual(dual.sin(-x), -dual.sin(x), self.format_param(x))

  def test_sin_asin_inv(self):
    for x, in self.sample():
      self.assert_inv(dual.sin, dual.asin, x)

  def test_cos_sym(self):
    for x, in self.sample():
      self.assertEqual(dual.cos(-x), dual.cos(x), self.format_param(x))

  def test_cos_acos_inv(self):
    for x, in self.sample():
      self.assert_inv(dual.cos, dual.acos, x)

  def test_tan_sym(self):
    for x, in self.sample():
      self.assertEqual(dual.tan(-x), -dual.tan(x), self.format_param(x))

  def test_tan_atan_inv(self):
    for x, in self.sample():
      self.assert_inv(dual.tan, dual.atan, x)

  def test_sin_cos_thm(self):
    for x, in self.sample():
      self.assertEqual(
        dual.cos(x)**2 + dual.sin(x)**2, 1, self.format_param(x))

  def test_sin_cos_tan_thm(self):
    for x, in self.sample():
      self.assertEqual(
        dual.sin(x) / dual.cos(x), dual.tan(x), self.format_param(x))

  def test_sinh_sym(self):
    for x, in self.sample():
      self.assertEqual(dual.sinh(-x), -dual.sinh(x), self.format_param(x))

  def test_sinh_asinh_inv(self):
    for x, in self.sample():
      self.assert_inv(dual.sinh, dual.asinh, x)

  def test_cosh_sym(self):
    for x, in self.sample():
      self.assertEqual(dual.cosh(-x), dual.cosh(x), self.format_param(x))

  def test_cosh_acosh_inv(self):
    for x, in self.sample():
      self.assert_inv(dual.cosh, dual.acosh, x)

  def test_tanh_sym(self):
    for x, in self.sample():
      self.assertEqual(dual.tanh(-x), -dual.tanh(x), self.format_param(x))

  def test_tanh_atanh_inv(self):
    for x, in self.sample():
      self.assert_inv(dual.tanh, dual.atanh, x)

  def test_sinh_cosh_thm(self):
    for x, in self.sample():
      self.assertEqual(
        dual.cosh(x)**2 - dual.sinh(x)**2, 1, self.format_param(x))

  def test_sinh_cosh_tanh_thm(self):
    for x, in self.sample():
      self.assertEqual(
        dual.sinh(x) / dual.cosh(x), dual.tanh(x), self.format_param(x))

  def assert_inv(self, f, i, x):
    def collapse_dual(x):
      return dual.Dual(self.collapse_scalar(x.a), x.b)
    y = f(x)
    if self.valid_for(i, y):
      self.assertEqual(collapse_dual(i(y)), x)
    if self.valid_for(i, x):
      self.assertEqual(collapse_dual(f(i(x))), x)

try:
  import sympy
except ImportError:
  has_sympy = False
else:
  has_sympy = True

@unittest.skipUnless(has_sympy, 'requires SymPy')
class DualSymbolTest(DualExactTest, unittest.TestCase):
  unit_count = 3
  max_pow    = 16

  @classmethod
  def setUpClass(cls):
    cls.duals = []
    term_count = 1 << cls.unit_count

    def make_dual(symbol):
      head, *tail = sympy.symbols('{}:{}'.format(symbol, term_count))
      return dual.Dual(head, dict(enumerate(tail, 1)))

    for symbol in 'abc':
      cls.duals.append(make_dual(symbol))

    cls.zero = make_dual('z')
    cls.zero.a = 0

  def setUp(self):
    dual.use_scalar('symbol')

  def tearDown(self):
    dual.use_scalar('real')

  def assertEqual(self, x, y, msg=None):
    x -= y
    x.a = sympy.simplify(x.a)
    x.b = {k: sympy.simplify(v) for k, v in x.b.items()}
    super().assertEqual(x, 0, msg)

  def test_pow_inv(self):
    for x, y in self.sample(2):
      if x.a != 0 and y.a != 0:
        p, _ = sympy.posify(x.a)
        x = dual.Dual(p, x.b)
        p, _ = sympy.posify(y.a)
        y = dual.Dual(p, y.b)
        self.assertEqual((x**y)**(1/y), x, self.format_param(x, y))
        self.assertEqual((x**(1/y))**y, x, self.format_param(x, y))

  def test_log_rcp(self):
    for x, in self.sample():
      if x.a != 0:
        p, _ = sympy.posify(x.a)
        x = dual.Dual(p, x.b)
        self.assertEqual(dual.log(1/x), -dual.log(x), self.format_param(x))

  def test_asin_log(self):
    for x, in self.sample():
      y = dual.asin(x)
      y.a = y.a.subs(
        sympy.asin(x.a), self.asin_to_log(sympy.sqrt, sympy.log, x.a))
      z = self.asin_to_log(dual.sqrt, dual.log, x)
      self.assertEqual(y, z, self.format_param(x))

  def test_acos_log(self):
    for x, in self.sample():
      y = dual.acos(x)
      y.a = y.a.subs(
        sympy.acos(x.a), self.acos_to_log(sympy.sqrt, sympy.log, x.a))
      z = self.acos_to_log(dual.sqrt, dual.log, x)
      self.assertEqual(y, z, self.format_param(x))

  def sample(self, n=1, *, allow_repeats=True):
    yield self.duals[:n]
    for i in range(n):
      yield [self.duals[j] if i != j else self.zero for j in range(n)]

  @staticmethod
  def valid_for(i, x):
    if i is dual.log:
      return x.a != 0

  @staticmethod
  def collapse_scalar(x):
    return sympy.simplify(x, inverse=True)

  @staticmethod
  def asin_to_log(sqrt, log, x):
    from sympy import I
    return -I * log(sqrt(1-x**2) + I*x)

  @staticmethod
  def acos_to_log(sqrt, log, x):
    from sympy import I
    return -I * log(I*sqrt(1-x**2) + x)

import math
import random
import sys

epsilon = sys.float_info.epsilon
sqrt_epsilon = math.sqrt(epsilon)

class DualNumberTest(DualTest):
  pure_count     = 4

  unit_count     = 32
  unit_zero_frac = 1/8

  mix_count      = 32
  max_fctr_count = 2
  max_term_count = 4
  mix_zero_frac  = 1/8

  @classmethod
  def setUpClass(cls):
    pures  = [cls.zero, cls.one]
    pures += [
      dual.Dual(cls.random(), {})
      for _ in range(cls.pure_count)]

    units = [
      dual.Dual.new(cls.random(), cls.random())
      for _ in range(cls.unit_count)]
    unit_keys = list(set(k for x in units for k in x.b.keys()))

    mixes = []
    for _ in range(cls.mix_count):
      while True:
        fctr_count = random.randint(1, cls.max_fctr_count)
        term_count = random.randint(1, cls.max_term_count)
        if fctr_count != 1 or term_count != 1:
          break
      mixes.append(dual.Dual(
        cls.random(),
        {functools.reduce(operator.or_, random.sample(unit_keys, fctr_count)):
        cls.random() for _ in range(term_count)}))

    for x in random.sample(units, round(cls.unit_count * cls.unit_zero_frac)):
      x.a = 0
    for x in random.sample(mixes, round(cls.mix_count * cls.mix_zero_frac)):
      x.a = 0

    cls.duals = pures + units + mixes

  def sample(self, n=1, *, allow_repeats=True):
    if allow_repeats:
      return itertools.product(self.duals, repeat=n)
    else:
      return itertools.combinations(self.duals, n)

class DualFloatTest(DualNumberTest):
  series_term_count = 32

  series_term_max = series_term_count * epsilon**(1/series_term_count) / math.e

  def assertAlmostEqual(self, x, y, msg=None):
    if not dual.isclose(x, y, abs_tol=sqrt_epsilon):
      std = '{!r} != {!r} in approximate sense'.format(x, y)
      msg = self._formatMessage(msg, std)
      raise self.failureException(msg)

  def test_exp_series(self):
    for x, in self.sample():
      self.assertAlmostEqual(
        dual.exp(x),
        sum(
          x**n / math.factorial(n)
          for n in range(self.series_term_count)),
        self.format_param(x))

  def test_sin_series(self):
    for x, in self.sample():
      self.assertAlmostEqual(
        dual.sin(x),
        sum(
          (-1 if n & 1 else 1) * x**(2*n+1) / math.factorial(2*n+1)
          for n in range(self.series_term_count)),
        self.format_param(x))

  def test_cos_series(self):
    for x, in self.sample():
      self.assertAlmostEqual(
        dual.cos(x),
        sum(
          (-1 if n & 1 else 1) * x**(2*n) / math.factorial(2*n)
          for n in range(self.series_term_count)),
        self.format_param(x))

  def test_sinh_series(self):
    for x, in self.sample():
      self.assertAlmostEqual(
        dual.sinh(x),
        sum(
          x**(2*n+1) / math.factorial(2*n+1)
          for n in range(self.series_term_count)),
        self.format_param(x))

  def test_cosh_series(self):
    for x, in self.sample():
      self.assertAlmostEqual(
        dual.cosh(x),
        sum(
          x**(2*n) / math.factorial(2*n)
          for n in range(self.series_term_count)),
        self.format_param(x))

class DualRealTest(DualFloatTest, unittest.TestCase):
  zero = dual.Dual(0, {})
  one  = dual.Dual(1, {})

  @classmethod
  def random(cls):
    return (
      2**random.uniform(
        math.log2(sqrt_epsilon), math.log2(cls.series_term_max)) *
      random.choice([-1, 1]))

class DualComplexTest(DualFloatTest, unittest.TestCase):
  zero = dual.Dual(0, {})
  one  = dual.Dual(1, {})

  def setUp(self):
    dual.use_scalar('complex')

  def tearDown(self):
    dual.use_scalar('real')

  @classmethod
  def random(cls):
    return complex(DualRealTest.random(), DualRealTest.random())

if __name__ == '__main__':
  unittest.main()
