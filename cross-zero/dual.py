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

def use_scalar(scalar):
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

use_scalar('real')

def drop_zeros(b):
  return {k: v for k, v in b.items() if v != 0}

class Dual:
  def __init__(self, a, b):
    self.a = a
    self.b = b

  next_token = 0

  @classmethod
  def token(cls):
    token = cls.next_token
    cls.next_token += 1
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
      if other == 0:
        return Dual(1, {})
      else:
        a = self.a ** other
        d = other * self.a ** (other-1)
        return Dual(a, {k: v * d for k, v in self.b.items()})
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
      return self.a == other.a and drop_zeros(self.b) == drop_zeros(other.b)
    elif isinstance(other, stype):
      return self.a == other and not drop_zeros(self.b)
    else:
      return NotImplemented

  def __round__(self, ndigits=None):
    a = round(self.a, ndigits)
    b = {k: round(v, ndigits) for k, v in self.b.items()}
    return Dual(a, drop_zeros(b))

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

def math_func(name, f, df):
  def result(x):
    if isinstance(x, Dual):
      d = df(x.a)
      return Dual(f(x.a), {k: v * d for k, v in x.b.items()})
    elif isinstance(x, stype):
      return f(x)
    else:
      raise TypeError(
        'must be {}, not {}'
        .format(format_types([Dual, *stype]), type(x).__name__))

  result.__name__ = result.__qualname__ = name
  return result

def reciprocal(x):
  try:
    return 1/x
  except ZeroDivisionError:
    raise ValueError('math domain error') from None

exp = math_func(
  'exp', lambda x: smath.exp(x), lambda x: smath.exp(x))
log = math_func(
  'log', lambda x: smath.log(x), reciprocal)
sin = math_func(
  'sin', lambda x: smath.sin(x), lambda x: smath.cos(x))
cos = math_func(
  'cos', lambda x: smath.cos(x), lambda x: -smath.sin(x))
tan = math_func(
  'tan', lambda x: smath.tan(x), lambda x: 1/smath.cos(x)**2)
asin = math_func(
  'asin', lambda x: smath.asin(x), lambda x: 1/(1-x**2)**sfrac(1, 2))
acos = math_func(
  'acos', lambda x: smath.acos(x), lambda x: -1/(1-x**2)**sfrac(1, 2))
atan = math_func(
  'atan', lambda x: smath.atan(x), lambda x: 1/(1+x**2))
sinh = math_func(
  'sinh', lambda x: smath.sinh(x), lambda x: smath.cosh(x))
cosh = math_func(
  'cosh', lambda x: smath.cosh(x), lambda x: smath.sinh(x))
tanh = math_func(
  'tanh', lambda x: smath.tanh(x), lambda x: 1/smath.cosh(x)**2)
asinh = math_func(
  'asinh', lambda x: smath.asinh(x), lambda x: 1/(1+x**2)**sfrac(1, 2))
acosh = math_func(
  'acosh', lambda x: smath.acosh(x), lambda x: 1/(x**2-1)**sfrac(1, 2))
atanh = math_func(
  'atanh', lambda x: smath.atanh(x), lambda x: 1/(1-x**2))

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
