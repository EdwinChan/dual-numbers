__all__ = [
  'Dual', 'isclose', 'sqrt', 'cbrt', 'hypot',
  'exp', 'expm1', 'log', 'log1p', 'log2', 'log10',
  'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
  'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh']

import collections
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

def drop_zeros(x):
  return {k: v for k, v in x.items() if v != 0}

class Dual(collections.UserDict):
  token = itertools.count()

  @classmethod
  def new(cls, a, v):
    return __class__({0: a, 1 << next(cls.token): v})

  def __pos__(self):
    return __class__(self)

  def __neg__(self):
    return __class__({k: -v for k, v in self.items()})

  def __add__(self, other):
    # pylint: disable-next=no-else-return
    if isinstance(other, __class__):
      return __class__(
        {**self, **{k: self.get(k, 0) + v for k, v in other.items()}})
    elif isinstance(other, stype):
      return __class__({**self, **{0: self.get(0, 0) + other}})
    else:
      return NotImplemented

  def __sub__(self, other):
    # pylint: disable-next=no-else-return
    if isinstance(other, __class__):
      return __class__(
        {**self, **{k: self.get(k, 0) - v for k, v in other.items()}})
    elif isinstance(other, stype):
      return __class__({**self, **{0: self.get(0, 0) - other}})
    else:
      return NotImplemented

  def __mul__(self, other):
    # pylint: disable-next=no-else-return
    if isinstance(other, __class__):
      x = __class__()
      for (k1, v1), (k2, v2) in itertools.product(self.items(), other.items()):
        if k1 & k2 == 0:
          k, v = k1 | k2, v1 * v2
          x[k] = x.get(k, 0) + v
      return x
    elif isinstance(other, stype):
      return __class__({k: v * other for k, v in self.items()})
    else:
      return NotImplemented

  def __truediv__(self, other):
    # pylint: disable-next=no-else-return
    if isinstance(other, __class__):
      return self * other**-1
    elif isinstance(other, stype):
      return __class__({k: v / other for k, v in self.items()})
    else:
      return NotImplemented

  def __pow__(self, other):
    # pylint: disable-next=no-else-return
    if isinstance(other, numbers.Integral):
      return pow_int(self, other)
    elif isinstance(other, stype + (__class__,)):
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
      return drop_zeros(self) == drop_zeros(other)
    elif isinstance(other, stype):
      return drop_zeros(self) == drop_zeros({0: other})
    else:
      return NotImplemented

  def __hash__(self):
    # pylint: disable-next=no-else-return
    if not (x := drop_zeros(self)) or x.keys() == {0}:
      return hash(x.get(0, 0))
    else:
      return hash(tuple(sorted(x.items())))

  def __round__(self, ndigits=None):
    return __class__(drop_zeros(
      {k: round(v, ndigits) for k, v in self.items()}))

  # pylint: disable-next=redefined-builtin
  def convert_to(self, type):
    if not (x := drop_zeros(self)) or x.keys() == {0}:
      try:
        return type(x.get(0, 0))
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

    tol = max(rel_tol * abs(self.get(0, 0)), abs_tol)

    return __class__({k: v for k, v in self.items() if abs(v) >= tol})

  def __repr__(self):
    return f'{__class__.__name__}({super().__repr__()})'

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
    return all(
      smath_isclose(first.get(k, 0), second.get(k, 0))
      for k in first.keys() | second.keys())
  elif first_dual and second_scalar:
    return dual_isclose(first, Dual({0: second}))
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
  #   fx_a      == f(x[0])
  #   fx_b_cfnz == lambda m: sum(math.perm(n, m) * c[n] * x[0]**n
  #                            for n in itertools.count())
  #   fx_b_cfz  == lambda m: math.factorial(m) * c[m]

  s = list(iter_set_bits(functools.reduce(operator.or_, x.keys(), 0)))
  m = len(s)
  s = list(iter_nonempty_subsets(s))

  k = list(map(sum, s))
  fx_b = dict.fromkeys(k, 0)

  a = x.get(0, 0)
  if a != 0:
    c = {k: x.get(k, 0) / a for k in k if k != 0}
    fx_b_cf = [fx_b_cfnz(m) for m in range(1, m+1)]
  else:
    c = {k: x.get(k, 0) for k in k if k != 0}
    fx_b_cf = [fx_b_cfz(m) for m in range(1, m+1)]

  for s in s:
    k = sum(s)
    for m in range(len(s)):
      q = fx_b_cf[m]
      for d in iter_stirling(s, m+1):
        d = map(sum, d)
        fx_b[k] += functools.reduce(operator.mul, (c[k] for k in d)) * q

  return Dual({**fx_b, **{0: fx_a}})

# integral power is derived by brute-force expansion
# other functions are defined through power-series expansion

def pow_int(x, n):
  if not isinstance(n, numbers.Integral):
    raise ValueError('can only raise to integer power')
  if x.get(0, 0) == 0 and n <= 0:
    raise ValueError('can only raise to positive integer power')
  a = x.get(0, 0)
  r = a**n
  # pylint: disable-next=no-else-return
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
      # pylint: disable-next=no-else-return
      if isinstance(x, Dual):
        return dual_func(x)
      elif isinstance(x, stype):
        return lazy_smath_func()(x)
      else:
        good_types = format_types([Dual, *stype])
        bad_type = format_types([type(x)])
        raise TypeError(f'must be {good_types}, not {bad_type}')
    return wrapper
  return wrap

@math_func(lambda: smath.exp)
def exp(x):
  a = x.get(0, 0)
  r = smath.exp(a)
  return func_from_series(
    x, r,
    lambda m: r * a**m,
    lambda m: 1)

@math_func(lambda: smath.expm1)
def expm1(x):
  y = exp(x)
  y[0] = smath.expm1(x.get(0, 0))
  return y

@math_func(lambda: smath.log)
def log(x):
  a = x.get(0, 0)
  r = smath.log(a)
  try:
    c = 1 - 1/a
  except ZeroDivisionError:
    raise ValueError('math domain error') from None
  return func_from_series(
    x-1, r,
    lambda m: (1 if m & 1 else -1) * math.factorial(m-1) * c**m,
    lambda m: (1 if m & 1 else -1) * math.factorial(m-1))

@math_func(lambda: smath.log1p)
def log1p(x):
  y = log(1+x)
  y[0] = smath.log1p(x.get(0, 0))
  return y

@math_func(lambda: smath.log2)
def log2(x):
  return log(x) / smath.log(2)

@math_func(lambda: smath.log10)
def log10(x):
  return log(x) / smath.log(10)

@math_func(lambda: smath.sin)
def sin(x):
  a = x.get(0, 0)
  c, s = smath.cos(a), smath.sin(a)
  return func_from_series(
    x, s,
    lambda m: (-1 if m & 2 else 1) * (c if m & 1 else s) * a**m,
    lambda m: (-1 if m & 2 else 1) * (m & 1))

@math_func(lambda: smath.cos)
def cos(x):
  a = x.get(0, 0)
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
  a = x.get(0, 0)
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
  y[0] = smath.acos(x.get(0, 0))
  return y

@math_func(lambda: smath.atan)
def atan(x):
  return asin(x / hypot(1, x))

@math_func(lambda: smath.sinh)
def sinh(x):
  a = x.get(0, 0)
  c, s = smath.cosh(a), smath.sinh(a)
  return func_from_series(
    x, smath.sinh(a),
    lambda m: (c if m & 1 else s) * a**m,
    lambda m: m & 1)

@math_func(lambda: smath.cosh)
def cosh(x):
  a = x.get(0, 0)
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
