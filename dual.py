__all__ = [
  'Dual', 'isclose', 'sqrt', 'cbrt', 'hypot', 'exp', 'log', 'sin', 'cos',
  'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh',
  'atanh']

import functools
import itertools
import math
import numbers
import operator
import sys

def set_scalar(scalar):
  global stype, sfrac, smath
  if scalar == 'real':
    stype = numbers.Real,
    sfrac = operator.truediv
    smath = math
  elif scalar == 'complex':
    import cmath
    stype = numbers.Complex,
    sfrac = operator.truediv
    smath = cmath
  elif scalar == 'symbol':
    import sympy
    stype = sympy.Basic, numbers.Number
    sfrac = sympy.Rational
    smath = sympy
  else:
    raise ValueError('unrecognized scalar')

set_scalar('real')

class Dual:
  def __init__(self, a, b):
    self.a = a
    self.b = b

  next_token = 1

  @classmethod
  def token(cls):
    token = cls.next_token
    cls.next_token <<= 1
    return token

  @classmethod
  def new(cls, a, v):
    return Dual(a, {cls.token(): v})

  def __pos__(self):
    return Dual(self.a, self.b.copy())

  def __neg__(self):
    return Dual(-self.a, {k: -v for k, v in self.b.items()})

  def __add__(self, other):
    if isinstance(other, Dual):
      return Dual(
        self.a + other.a,
        {**self.b, **{k: self.b.get(k, 0) + v for k, v in other.b.items()}})
    elif isinstance(other, stype):
      return Dual(self.a + other, self.b)
    else:
      return NotImplemented

  def __sub__(self, other):
    if isinstance(other, Dual):
      return Dual(
        self.a - other.a,
        {**self.b, **{k: self.b.get(k, 0) - v for k, v in other.b.items()}})
    elif isinstance(other, stype):
      return Dual(self.a - other, self.b)
    else:
      return NotImplemented

  def __mul__(self, other):
    if isinstance(other, Dual):
      a = self.a * other.a
      b = {}
      for k, v in self.b.items():
        b[k] = b.get(k, 0) + v * other.a
      for k, v in other.b.items():
        b[k] = b.get(k, 0) + v * self.a
      for (k1, v1), (k2, v2) in itertools.product(
          self.b.items(), other.b.items()):
        if k1 & k2 == 0:
          k, v = k1 | k2, v1 * v2
          b[k] = b.get(k, 0) + v
      return Dual(a, b)
    elif isinstance(other, stype):
      return Dual(self.a * other, {k: v * other for k, v in self.b.items()})
    else:
      return NotImplemented

  def __truediv__(self, other):
    if isinstance(other, Dual):
      return self * other**-1
    elif isinstance(other, stype):
      return Dual(self.a / other, {k: v / other for k, v in self.b.items()})
    else:
      return NotImplemented

  def __pow__(self, other):
    if isinstance(other, numbers.Integral):
      return pow_int(self, other)
    elif isinstance(other, stype + (Dual,)):
      return exp(other * log(self))
    else:
      return NotImplemented

  def __radd__(self, other):
    if isinstance(other, stype):
      return self + other
    else:
      return NotImplemented

  def __rsub__(self, other):
    if isinstance(other, stype):
      return -self + other
    else:
      return NotImplemented

  def __rmul__(self, other):
    if isinstance(other, stype):
      return self * other
    else:
      return NotImplemented

  def __rtruediv__(self, other):
    if isinstance(other, stype):
      return self**-1 * other
    else:
      return NotImplemented

  def __rpow__(self, other):
    if isinstance(other, stype):
      return exp(self * smath.log(other))
    else:
      return NotImplemented

  def __eq__(self, other):
    if isinstance(other, Dual):
      c = {k: v for k, v in self.b.items() if v != 0}
      d = {k: v for k, v in other.b.items() if v != 0}
      return self.a == other.a and c == d
    elif isinstance(other, stype):
      return self.a == other and all(v == 0 for v in self.b.values())
    else:
      return NotImplemented

  def __round__(self, ndigits=None):
    a = round(self.a, ndigits)
    b = {k: round(v, ndigits) for k, v in self.b.items()}
    b = {k: v for k, v in b.items() if v != 0}
    return Dual(a, b)

  def chop(self, *, rel_tol=1e-9, abs_tol=0):
    if rel_tol < 0 or abs_tol < 0:
      raise ValueError('tolerances must be non-negative')

    tol = max(rel_tol * abs(self.a), abs_tol)
    def is_finite(x):
      return abs(x) >= tol

    return Dual(
      self.a if is_finite(self.a) else 0,
      {k: v for k, v in self.b.items() if is_finite(v)})

  def __repr__(self):
    return 'Dual({}, {})'.format(self.a, self.b)

def isclose(first, second, *, rel_tol=1e-9, abs_tol=0):
  if not hasattr(smath, 'isclose'):
    raise ValueError('scalar supports only exact operations')
  if rel_tol < 0 or abs_tol < 0:
    raise ValueError('tolerances must be non-negative')

  first_dual    = isinstance(first, Dual)
  first_scalar  = isinstance(first, stype)
  second_dual   = isinstance(second, Dual)
  second_scalar = isinstance(second, stype)

  smath_isclose = functools.partial(
    smath.isclose, rel_tol=rel_tol, abs_tol=abs_tol)

  if first_dual and second_dual:
    return (
      smath_isclose(first.a, second.a) and
      all(smath_isclose(v, second.b.get(k, 0)) for k, v in first.b.items()) and
      all(smath_isclose(v, first.b.get(k, 0)) for k, v in second.b.items()))
  elif first_dual and second_scalar:
    return (
      smath_isclose(first.a, second) and
      all(smath_isclose(v, 0) for k, v in first.b.items()))
  elif first_scalar and second_dual:
    return isclose(second, first)
  elif first_scalar and second_scalar:
    return smath_isclose(first, second)
  elif not (first_dual or first_scalar):
    raise TypeError(
      'must be {}, not {}'
      .format(format_types([Dual, *stype]), type(first).__name__))
  elif not (second_dual or second_scalar):
    raise TypeError(
      'must be {}, not {}'
      .format(format_types([Dual, *stype]), type(second).__name__))

def sqrt(x):
  return x**sfrac(1, 2)

def cbrt(x):
  return x**sfrac(1, 3)

def hypot(*x):
  return sqrt(sum(x**2 for x in x))

def iter_set_bits(n):
  p = 1
  while n > 0:
    if n & 1:
      yield p
    n >>= 1
    p <<= 1

def iter_nonempty_subsets(x):
  x = list(x)
  for n in range(1, len(x)+1):
    yield from itertools.combinations(x, n)

def iter_stirling(x, k):
  # [[https://devblogs.microsoft.com/oldnewthing/20140324-00/?p=1413]]
  if not x and k <= 0:
    yield []
  elif x and k > 0:
    h, t = x[0], x[1:]
    for d in iter_stirling(t, k-1):
      yield [[h]] + d
    for d in iter_stirling(t, k):
      for i in range(k):
        yield d[:i] + [[h] + d[i]] + d[i+1:]

def func_from_series(x, fx_a, fx_b_cfnz, fx_b_cfz):
  # this function defines a mathematical function f(x) through its power-series
  # expansion around x == 0:
  #   f(x) == sum(c[n] * x**n for n in itertools.count())
  # where c[n] is the nth derivative of f evaluated at x == 0 divided by
  # math.factorial(n)

  # the code makes use of the structure common to all such power-series
  # expansions; because of this, the function parameters do not have intuitive
  # explanations, but are instead defined mathematically through
  #   fx_a      == f(x.a)
  #   fx_b_cfnz == lambda m: sum(math.perm(n, m) * c[n] * x.a**n
  #                            for n in itertools.count())
  #   fx_b_cfz  == lambda m: math.factorial(m) * c[m]

  s = list(iter_set_bits(functools.reduce(operator.or_, x.b.keys(), 0)))
  m = len(s)
  s = list(iter_nonempty_subsets(s))

  k = list(map(sum, s))
  fx_b = dict.fromkeys(k, 0)

  if x.a != 0:
    c = {k: x.b.get(k, 0) / x.a for k in k}
    fx_b_cf = [fx_b_cfnz(m) for m in range(1, m+1)]
  else:
    c = {k: x.b.get(k, 0) for k in k}
    fx_b_cf = [fx_b_cfz(m) for m in range(1, m+1)]

  for s in s:
    k = sum(s)
    for m in range(len(s)):
      q = fx_b_cf[m]
      for d in iter_stirling(s, m+1):
        d = map(sum, d)
        fx_b[k] += functools.reduce(operator.mul, (c[k] for k in d)) * q

  return Dual(fx_a, fx_b)

# integral power is derived by brute-force expansion
# other functions are defined through power-series expansion

def pow_int(x, n):
  if not isinstance(n, numbers.Integral):
    raise ValueError('can only raise to integer power')
  if x.a == 0 and n < 0:
    raise ZeroDivisionError('can only raise to non-negative integer power')
  a = x.a
  r = a**n
  if n >= 0:
    return func_from_series(
      x, r,
      lambda m: r * math.perm(n, m),
      lambda m: math.factorial(n) if m == n else 0)
  else:
    # lambda expression can be obtained by taking limit of math.perm(n, m) =
    # math.gamma(n+1) / math.gamma(n-m+1) at n < 0
    return func_from_series(
      x, r,
      lambda m: r * (-1 if m & 1 else 1) * math.perm(m-n-1, m),
      None)

def double_factorial(n):
  return functools.reduce(operator.mul, range(n, 0, -2), 1)

def math_func(lazy_smath_func):
  def wrap(dual_func):
    def wrapper(x):
      if isinstance(x, Dual):
        return dual_func(x)
      elif isinstance(x, stype):
        return lazy_smath_func()(x)
      else:
        raise TypeError(
          'must be {}, not {}'
          .format(format_types([Dual, *stype]), type(x).__name__))
    return wrapper
  return wrap

@math_func(lambda: smath.exp)
def exp(x):
  a = x.a
  r = smath.exp(a)
  return func_from_series(
    x, r,
    lambda m: r * a**m,
    lambda m: 1)

@math_func(lambda: smath.log)
def log(x):
  a = x.a
  if a == 0:
    raise ValueError('math domain error')
  r = smath.log(a)
  c = 1 - 1/a
  return func_from_series(
    x-1, r,
    lambda m: (1 if m & 1 else -1) * math.factorial(m-1) * c**m,
    lambda m: (1 if m & 1 else -1) * math.factorial(m-1))

@math_func(lambda: smath.sin)
def sin(x):
  a = x.a
  c, s = smath.cos(a), smath.sin(a)
  return func_from_series(
    x, s,
    lambda m: (-1 if m & 2 else 1) * (c if m & 1 else s) * a**m,
    lambda m: (-1 if m & 2 else 1) * (m & 1))

@math_func(lambda: smath.cos)
def cos(x):
  a = x.a
  c, s = smath.cos(a), smath.sin(a)
  return func_from_series(
    x, c,
    lambda m: (-1 if m+1 & 2 else 1) * (s if m & 1 else c) * a**m,
    lambda m: (-1 if m+1 & 2 else 1) * (m+1 & 1))

@math_func(lambda: smath.tan)
def tan(x):
  return sin(x) / cos(x)

@math_func(lambda: smath.asin)
def asin(x):
  a = x.a
  return func_from_series(
    x, smath.asin(a),
    lambda m:
      math.factorial(m-1)**2 * a**(m*2-1) / (1-a**2)**(m-sfrac(1, 2)) *
      sum((a*2)**-l / math.factorial(m-l-1) / math.factorial(l>>1)**2
        for l in range(0, m, 2)),
    lambda m: double_factorial(m-2)**2 if m & 1 else 0)

@math_func(lambda: smath.acos)
def acos(x):
  # smath.pi/2 - asin(x)
  y = -asin(x)
  y.a = smath.acos(x.a)
  return y

@math_func(lambda: smath.atan)
def atan(x):
  return asin(x / hypot(1, x))

@math_func(lambda: smath.sinh)
def sinh(x):
  a = x.a
  c, s = smath.cosh(a), smath.sinh(a)
  return func_from_series(
    x, smath.sinh(a),
    lambda m: (c if m & 1 else s) * a**m,
    lambda m: m & 1)

@math_func(lambda: smath.cosh)
def cosh(x):
  a = x.a
  c, s = smath.cosh(a), smath.sinh(a)
  return func_from_series(
    x, smath.cosh(a),
    lambda m: (s if m & 1 else c) * a**m,
    lambda m: m+1 & 1)

@math_func(lambda: smath.tanh)
def tanh(x):
  return sinh(x) / cosh(x)

@math_func(lambda: smath.asinh)
def asinh(x):
  return log(x + hypot(1, x))

@math_func(lambda: smath.acosh)
def acosh(x):
  return log(x + sqrt(x-1) * sqrt(x+1))

@math_func(lambda: smath.atanh)
def atanh(x):
  return (log(1+x) - log(1-x)) / 2

def format_alts(alts):
  alts = list(map(str, alts))
  if len(alts) == 0:
    return ''
  elif len(alts) == 1:
    return alts[0]
  elif len(alts) == 2:
    return '{} or {}'.format(*alts)
  else:
    return ', '.join(alts[:-1]) + ', or ' + alts[-1]

def format_types(types):
  return format_alts(type.__name__ for type in types)
