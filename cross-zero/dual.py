__all__ = [
  'Dual', 'isclose', 'sqrt', 'cbrt', 'hypot',
  'exp', 'expm1', 'log', 'log1p', 'log2', 'log10',
  'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
  'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh']

import functools
import itertools
import math
import numbers
import operator

def use_scalar(scalar):
  # pylint: disable-next=global-variable-undefined
  global stype, sfrac, smath
  if scalar == 'real':
    stype = (numbers.Real,)
    sfrac = operator.truediv
    smath = math
  elif scalar == 'complex':
    # pylint: disable-next=import-outside-toplevel
    import cmath
    stype = (numbers.Complex,)
    sfrac = operator.truediv
    smath = cmath
  elif scalar == 'symbol':
    # pylint: disable=import-outside-toplevel
    import types
    import sympy
    # pylint: enable=import-outside-toplevel
    stype = (sympy.Basic, numbers.Number)
    sfrac = sympy.Rational
    smath = types.ModuleType('sympy')
    for key, value in vars(sympy).items():
      setattr(smath, key, value)
    def sympy_log(x):
      # pylint: disable-next=no-else-raise
      if x == 0:
        raise ValueError('math domain error')
      else:
        return sympy.log(x)
    smath.log   = sympy_log
    # pylint: disable=no-member
    smath.expm1 = lambda x: smath.exp(x) - 1
    smath.log1p = lambda x: smath.log(1+x)
    smath.log2  = lambda x: smath.log(x) / smath.log(2)
    smath.log10 = lambda x: smath.log(x) / smath.log(10)
    # pylint: enable=no-member
  else:
    raise ValueError('unrecognized scalar')

use_scalar('real')

def drop_zeros(b):
  return {k: v for k, v in b.items() if v != 0}

class Dual:
  def __init__(self, a, b):
    self.a = a
    self.b = b

  token = itertools.count()

  @classmethod
  def new(cls, a, v):
    return __class__(a, {next(cls.token): v})

  def __pos__(self):
    return __class__(self.a, self.b.copy())

  def __neg__(self):
    return __class__(-self.a, {k: -v for k, v in self.b.items()})

  def __add__(self, other):
    # pylint: disable-next=no-else-return
    if isinstance(other, __class__):
      return __class__(
        self.a + other.a,
        {**self.b, **{k: self.b.get(k, 0) + v for k, v in other.b.items()}})
    elif isinstance(other, stype):
      return __class__(self.a + other, self.b)
    else:
      return NotImplemented

  def __sub__(self, other):
    # pylint: disable-next=no-else-return
    if isinstance(other, __class__):
      return __class__(
        self.a - other.a,
        {**self.b, **{k: self.b.get(k, 0) - v for k, v in other.b.items()}})
    elif isinstance(other, stype):
      return __class__(self.a - other, self.b)
    else:
      return NotImplemented

  def __mul__(self, other):
    # pylint: disable-next=no-else-return
    if isinstance(other, __class__):
      a = self.a * other.a
      b = {}
      for k, v in self.b.items():
        b[k] = b.get(k, 0) + v * other.a
      for k, v in other.b.items():
        b[k] = b.get(k, 0) + v * self.a
      return __class__(a, b)
    elif isinstance(other, stype):
      return __class__(
        self.a * other, {k: v * other for k, v in self.b.items()})
    else:
      return NotImplemented

  def __truediv__(self, other):
    # pylint: disable-next=no-else-return
    if isinstance(other, __class__):
      return self * other**-1
    elif isinstance(other, stype):
      return __class__(
        self.a / other, {k: v / other for k, v in self.b.items()})
    else:
      return NotImplemented

  def __pow__(self, other):
    # pylint: disable-next=no-else-return
    if isinstance(other, stype):
      try:
        a = self.a ** other
        d = other * self.a ** (other-1)
      except ZeroDivisionError:
        raise ValueError('math domain error') from None
      return __class__(a, {k: v * d for k, v in self.b.items()})
    elif isinstance(other, __class__):
      return exp(other * log(self))
    else:
      return NotImplemented

  def __radd__(self, other):
    return self + other

  def __rsub__(self, other):
    return -self + other

  def __rmul__(self, other):
    return self * other

  def __rtruediv__(self, other):
    return self**-1 * other

  def __rpow__(self, other):
    return exp(self * smath.log(other))

  def __eq__(self, other):
    # pylint: disable-next=no-else-return
    if isinstance(other, __class__):
      return self.a == other.a and drop_zeros(self.b) == drop_zeros(other.b)
    elif isinstance(other, stype):
      return self.a == other and not drop_zeros(self.b)
    else:
      return NotImplemented

  def __hash__(self):
    # pylint: disable-next=no-else-return
    if not (b := drop_zeros(self.b)):
      return hash(self.a)
    else:
      return hash((self.a, tuple(sorted(b.items()))))

  def __round__(self, ndigits=None):
    a = round(self.a, ndigits)
    b = {k: round(v, ndigits) for k, v in self.b.items()}
    return __class__(a, drop_zeros(b))

  # pylint: disable-next=redefined-builtin
  def convert_to(self, type):
    if not drop_zeros(self.b):
      try:
        return type(self.a)
      except TypeError:
        pass
    raise ValueError(f'cannot convert to {type}')

  def __int__(self):
    return self.convert_to(int)

  def __float__(self):
    return self.convert_to(float)

  def __complex__(self):
    return self.convert_to(complex)

  def chop(self, *, rel_tol=1e-9, abs_tol=0):
    if rel_tol < 0 or abs_tol < 0:
      raise ValueError('tolerances must be non-negative')

    tol = max(rel_tol * abs(self.a), abs_tol)
    def is_finite(x):
      return abs(x) >= tol

    return __class__(
      self.a if is_finite(self.a) else 0,
      {k: v for k, v in self.b.items() if is_finite(v)})

  def __repr__(self):
    return f'{__class__.__name__}({self.a!r}, {self.b!r})'

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
  dual_isclose = functools.partial(
    isclose, rel_tol=rel_tol, abs_tol=abs_tol)

  # pylint: disable-next=no-else-return
  if first_dual and second_dual:
    return (
      smath_isclose(first.a, second.a) and
      all(
        smath_isclose(first.b.get(k, 0), second.b.get(k, 0))
        for k in first.b.keys() | second.b.keys()))
  elif first_dual and second_scalar:
    return dual_isclose(first, Dual(second, {}))
  elif first_scalar and second_dual:
    # pylint: disable-next=arguments-out-of-order
    return dual_isclose(second, first)
  elif first_scalar and second_scalar:
    return smath_isclose(first, second)
  elif not (first_dual or first_scalar):
    good_types = format_types([Dual, *stype])
    bad_type = format_types([type(first)])
    raise TypeError(f'must be {good_types}, not {bad_type}')
  elif not (second_dual or second_scalar):
    good_types = format_types([Dual, *stype])
    bad_type = format_types([type(second)])
    raise TypeError(f'must be {good_types}, not {bad_type}')
  else:
    raise ValueError('cannot happen')

def sqrt(x):
  return x**sfrac(1, 2)

def cbrt(x):
  return x**sfrac(1, 3)

def hypot(*x):
  return sqrt(sum(x**2 for x in x))

def math_func(name, f, df):
  def result(x):
    # pylint: disable-next=no-else-return
    if isinstance(x, Dual):
      d = df(x.a)
      return Dual(f(x.a), {k: v * d for k, v in x.b.items()})
    elif isinstance(x, stype):
      return f(x)
    else:
      good_types = format_types([Dual, *stype])
      bad_type = format_types([type(x)])
      raise TypeError(f'must be {good_types}, not {bad_type}')

  result.__name__ = result.__qualname__ = name
  return result

def reciprocal(x):
  try:
    return 1/x
  except ZeroDivisionError:
    raise ValueError('math domain error') from None

# lambdas use current value of smath
# pylint: disable=unnecessary-lambda
exp = math_func(
  'exp', lambda x: smath.exp(x), lambda x: smath.exp(x))
expm1 = math_func(
  'expm1', lambda x: smath.expm1(x), lambda x: smath.exp(x))
log = math_func(
  'log', lambda x: smath.log(x), reciprocal)
log1p = math_func(
  'log1p', lambda x: smath.log1p(x), lambda x: reciprocal(1+x))
log2 = math_func(
  'log2', lambda x: smath.log2(x), lambda x: reciprocal(x) / smath.log(2))
log10 = math_func(
  'log10', lambda x: smath.log10(x), lambda x: reciprocal(x) / smath.log(10))
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
# pylint: enable=unnecessary-lambda

def format_alts(alts):
  alts = list(map(str, alts))
  # pylint: disable-next=no-else-return
  if len(alts) == 0:
    return ''
  elif len(alts) == 1:
    return alts[0]
  elif len(alts) == 2:
    # pylint: disable-next=consider-using-f-string
    return '{} or {}'.format(*alts)
  else:
    return ', '.join(alts[:-1]) + ', or ' + alts[-1]

def format_types(types):
  return format_alts(type.__name__ for type in types)
