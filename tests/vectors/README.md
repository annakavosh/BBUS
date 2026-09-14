# Shared Bayes vectors

`bayes_vectors.csv` is the single deterministic fixture consumed by the native
C++ test and the Python equivalence test. Expected calculations were prepared
independently by hand and include an explicit absolute and relative tolerance
on every row.

Valid rows exercise exact-binary and log-odds updates. Invalid rows use `-` for
unavailable expected outputs and must be rejected before a calculation is
returned. The pure-Python implementation under `tests/python/` exists only as a
verification oracle; production Bayesian logic remains in C++.
