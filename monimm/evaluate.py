"""Evaluation of monomial immanants, in floating point (batched) and exactly.

For a class function phi on S_n and an n x n matrix A,

    d_phi(A) = sum_{sigma in S_n} phi(sigma) prod_i A[i, sigma(i)].

Since phi is constant on conjugacy classes, group the sum by cycle type:

    d_phi(A) = sum_rho phi(rho) * S_rho(A),
    S_rho(A) = sum_{sigma of type rho} prod_i A[i, sigma(i)].

The p(n) "class sums" S_rho(A) are computed once per matrix, and then every
monomial immanant is a single dot product ``Imm_lambda(A) = G[lambda] . S(A)``
with the exact table ``G`` from :func:`combinatorics.monomial_class_functions`.
The permanent is ``sum_rho S_rho(A)``.

Floating-point safety.  If A is entrywise non-negative (true for TNN
matrices) then every ``S_rho(A) >= 0``, and the rounding error in
``G[lambda] . S(A)`` is bounded by a small multiple of machine epsilon times
``sum_rho |G[lambda][rho]| S_rho(A)``.  :meth:`ImmanantEvaluator.error_bound`
returns that quantity; a computed value is only treated as a genuine negative
if it is below minus this bound, and even then it is re-checked exactly.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations
from typing import List, Sequence, Tuple

import numpy as np

from .combinatorics import Partition, cycle_type, monomial_class_functions


class ImmanantEvaluator:
    """Precomputed data for evaluating all monomial immanants of n x n matrices.

    Attributes
    ----------
    n : int
    parts : tuple of partitions of n (reverse lexicographic order)
    perms : int array of shape (n!, n), all permutations in one-line notation
    class_of_perm : int array of shape (n!,), index into ``parts`` of the
        cycle type of each permutation
    class_indicator : float array (n!, p(n)), one-hot of ``class_of_perm``
    g_exact : tuple of tuples of Fraction, ``g_exact[a][b] = g_{parts[a]}(parts[b])``
    g_float : float array (p(n), p(n)), the same table as floats
    g_abs : ``abs(g_float)``, used for the rounding-error bound
    """

    def __init__(self, n: int):
        self.n = n
        self.parts, self.g_exact = monomial_class_functions(n)
        self.index_of = {mu: i for i, mu in enumerate(self.parts)}

        # All n! permutations and the conjugacy class of each one.
        perm_list = list(permutations(range(n)))
        self.perms = np.array(perm_list, dtype=np.intp)
        self.class_of_perm = np.array(
            [self.index_of[cycle_type(p)] for p in perm_list], dtype=np.intp)

        # One-hot matrix so that (products over perms) @ indicator = class sums.
        self.class_indicator = np.zeros((len(perm_list), len(self.parts)))
        self.class_indicator[np.arange(len(perm_list)), self.class_of_perm] = 1.0

        self.g_float = np.array([[float(x) for x in row] for row in self.g_exact])
        self.g_abs = np.abs(self.g_float)

    # -- floating point ----------------------------------------------------

    def class_sums(self, mats: np.ndarray) -> np.ndarray:
        """Return ``S[b, r] = S_{parts[r]}(mats[b])`` for a batch of matrices.

        ``mats`` has shape (B, n, n) (a single (n, n) matrix is also accepted).
        Memory use is O(B * n!): the product ``prod_i A[i, sigma(i)]`` is
        accumulated one row at a time instead of materialising a (B, n!, n)
        array.
        """
        mats = np.asarray(mats, dtype=float)
        if mats.ndim == 2:
            mats = mats[None]
        products = np.ones((mats.shape[0], self.perms.shape[0]))
        for i in range(self.n):
            # mats[:, i, perms[:, i]] has shape (B, n!): entry A[i, sigma(i)].
            products *= mats[:, i, self.perms[:, i]]
        return products @ self.class_indicator

    def monomial_from_sums(self, sums: np.ndarray) -> np.ndarray:
        """Return ``Imm[b, a] = Imm_{parts[a]}`` given class sums ``S[b, :]``."""
        return sums @ self.g_float.T

    def error_bound(self, sums: np.ndarray) -> np.ndarray:
        """Return a rounding-error scale for each ``Imm[b, a]``.

        ``64 * eps * sum_r |g_a(r)| S_r`` -- a deliberately generous bound for
        the dot product, valid because all ``S_r >= 0`` for entrywise
        non-negative matrices.  (The class sums themselves are products of
        n non-negative numbers summed with non-negative terms only, so they
        carry only a small relative error.)
        """
        return 64 * np.finfo(float).eps * (sums @ self.g_abs.T)

    def monomial(self, mats: np.ndarray) -> np.ndarray:
        """Return all monomial immanants, shape (B, p(n)), for a batch of matrices."""
        return self.monomial_from_sums(self.class_sums(mats))

    def permanent(self, mats: np.ndarray) -> np.ndarray:
        """Return the permanent of each matrix in the batch, shape (B,)."""
        return self.class_sums(mats).sum(axis=1)

    # -- exact ---------------------------------------------------------------

    def class_sums_exact(self, mat: Sequence[Sequence]) -> List[Fraction]:
        """Exact class sums ``S_rho(A)`` for one matrix with Fraction/int entries."""
        sums = [Fraction(0)] * len(self.parts)
        for perm, cls in zip(self.perms.tolist(), self.class_of_perm.tolist()):
            term = Fraction(1)
            for i, j in enumerate(perm):
                term *= mat[i][j]
                if term == 0:
                    break
            sums[cls] += term
        return sums

    def monomial_exact(self, mat: Sequence[Sequence]) -> List[Fraction]:
        """Exact values ``[Imm_lambda(A) for lambda in parts]`` for one matrix."""
        sums = self.class_sums_exact(mat)
        return [sum((g * s for g, s in zip(row, sums)), Fraction(0))
                for row in self.g_exact]

    def class_function_exact(self, phi: Sequence, mat: Sequence[Sequence]) -> Fraction:
        """Exact ``d_phi(A)`` for a class function given by values ``phi[rho]``.

        ``phi`` is indexed like ``parts`` (value of phi on each cycle type).
        """
        sums = self.class_sums_exact(mat)
        return sum((Fraction(p) * s for p, s in zip(phi, sums)), Fraction(0))

    def index(self, mu: Partition) -> int:
        """Position of partition ``mu`` in ``parts``."""
        return self.index_of[tuple(mu)]

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"ImmanantEvaluator(n={self.n}, p(n)={len(self.parts)})"


def split_params(x: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split a flat parameter array (..., n(n-1)) into (lower, upper) halves."""
    half = n * (n - 1) // 2
    return x[..., :half], x[..., half:]
