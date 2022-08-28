This Python module implements dual numbers of the form a + bi + cj + dk + …,
where a, b, c, d, … are elements of some underlying field, typically the field
of real numbers, and i, j, k, … are dual units. The multiplication of dual
units is associative and commutative, and the product of dual units vanishes.
Dual numbers satisfy the properties of a field.

## Applications

Dual numbers model the values and first derivatives of functions at a certain
point. For example, the value of the function f(p, q) and its first derivatives
can be combined into a single dual number as

x = f + (∂f/∂p) i + (∂f/∂q) j

We can do the same for the function g(p, q):

y = g + (∂g/∂p) i + (∂g/∂q) j

The product xy of the two dual numbers contains information about fg and its
first derivatives:

  xy  
= fg + [f (∂g/∂p) + g (∂f/∂p)] i + [f (∂g/∂q) + g (∂f/∂q)] j  
= fg + [∂(fg)/∂p] i + [∂(fg)/∂q] j

In addition to functions and first derivatives, dual numbers can also represent
measurements with uncertainties because uncertainties, like derivatives, are
first-order corrections. Each dual unit signifies one source of uncertainty;
for example, the dual number a + bi + cj denotes a measurement with two
distinct sources of uncertainty. The commutative nature of multiplication
reflects the fact that the uncertainty associated with the product of two
measurements is independent of the order in which the two measurements are
multiplied together.

## Properties

Any elementary function on dual numbers can be defined in terms of the
power-series expansion of the same function on the underlying field. It follows
immediately that for any function f(x), we have

f(a + bi + cj + dk + …) = f(a) + f'(a) (bi + cj + dk + …)

A function on dual numbers is undefined at the dual number a + … if the
derivative of the same function on the underlying field is undefined at a.

## Implementation

Dual units can be encoded as i ↦ 0, j ↦ 1, k ↦ 2, and so on.

## Examples

The dual numbers x = 1 + 2i and y = 3 + 4j are created by

```python
>>> from dual import Dual
>>> x = Dual.new(1, 2)
>>> y = Dual.new(3, 4)
>>> x
Dual(1, {0: 2})
>>> y
Dual(3, {1: 4})
```

Notice that we used `Dual.new()` instead of the `Dual` constructor.
`Dual.new()` generates a new dual unit every time it is called, so the dual
numbers it returns have distinct dual units; that is to say, we have y = 3 + 4j
instead of y = 3 + 4i.

The `Dual` constructor allows us to specify the dual units explicitly. The
second argument to the constructor is a `dict` whose keys are encoded dual
units and whose values are the corresponding coefficients. The constructor is
needed if we want to reuse a dual unit, as in z = 5 + 6i:

```python
>>> z = Dual(5, {0: 6})
>>> z
Dual(5, {0: 6})
```

We can perform arithmetic operations on dual numbers:

```python
>>> x**2
Dual(1, {0: 4})
>>> y**2
Dual(9, {1: 24})
>>> x*y
Dual(3, {0: 6, 1: 4})
```

This tells us that x² = 1 + 4i, y² = 9 + 24j, and xy = 3 + 6i + 4j.
