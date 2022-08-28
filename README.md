This Python module implements dual numbers of the form a + bi + cj + dk + …,
where a, b, c, d, … are elements of some underlying field, typically the field
of real numbers, and i, j, k, … are dual units.

There are two flavors of the module, differing in how they treat cross terms
that contain products of different dual units, such as ij. The version in
`cross-zero` drops cross terms and the one in `cross-nonzero` keeps them.
