"""Totally non-negative matrices via the bidiagonal (Loewner-Whitney) factorization.

Background
----------
Write ``L_k(t) = I + t E_{k+1,k}`` and ``U_k(t) = I + t E_{k,k+1}`` for the
elementary lower / upper bidiagonal matrices (0-indexed here, k = 0..n-2).
For t >= 0 these are totally non-negative (TNN).

The Loewner-Whitney theorem (see Fallat-Johnson, *Totally Nonnegative
Matrices*, Ch. 2) says that every n x n totally positive (TP) matrix factors
uniquely as

    A = L_{w_1}(t_1) ... L_{w_N}(t_N) * D * U_{v_1}(s_1) ... U_{v_N}(s_N)

with N = n(n-1)/2, all t_i, s_i > 0, D a positive diagonal matrix, and where
``w = (w_1, ..., w_N)`` is a fixed reduced word for the longest permutation
w_0 of S_n (``v`` is the reversed word).  Conversely every such product with
positive parameters is TP.  Allowing parameters to be 0 gives TNN matrices
on the boundary, and by Whitney's theorem TNN_n is the closure of TP_n.  So sampling the parameters in [0, inf) explores
all of TNN_n up to closure, which is what the search code needs.

The reduced word used for the lower part is

    (n-2) | (n-3, n-2) | (n-4, n-3, n-2) | ... | (0, 1, ..., n-2)

(0-indexed), the standard choice in Fallat-Johnson.  For n = 3 this is
``s_1 s_0 s_1``, a reduced word for w_0.

Diagonal factor.  For any class function phi and diagonal D,
``d_phi(D A) = det(D) d_phi(A)``, so D only rescales every immanant by the
same positive number and does not affect signs or ratios.  The search code
therefore uses D = I, but D is supported here for completeness.

Two implementations are given: a vectorised NumPy one operating on a batch of
parameter vectors (fast, floating point), and an exact one using Fractions
(slow, used to certify any suspected counterexample).
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import List, Optional, Sequence

import numpy as np

from .combinatorics import Partition


def lower_word(n: int) -> List[int]:
    """Return the reduced word for w_0 used for the lower bidiagonal factors.

    Entry ``k`` stands for ``L_k(t) = I + t E_{k+1,k}`` (0-indexed).  The
    word has length n(n-1)/2.
    """
    word: List[int] = []
    # Block j (j = 1..n-1) is (n-1-j, n-j, ..., n-2).
    for j in range(1, n):
        word.extend(range(n - 1 - j, n - 1))
    return word


def upper_word(n: int) -> List[int]:
    """Return the word for the upper factors: the lower word reversed.

    With this choice ``U_{v_1}(s_1) ... U_{v_N}(s_N)`` is the transpose of
    ``L_{w_1}(s_N) ... L_{w_N}(s_1)``, so the upper part is again a generic
    upper-unitriangular TNN matrix.
    """
    return lower_word(n)[::-1]


def num_params(n: int) -> int:
    """Number of parameters in each of the lower and upper parts: n(n-1)/2."""
    return n * (n - 1) // 2


def build_tnn_batch(lower_t: np.ndarray, upper_t: np.ndarray,
                    diag: Optional[np.ndarray] = None) -> np.ndarray:
    """Build a batch of TNN matrices from bidiagonal-factorization parameters.

    Parameters
    ----------
    lower_t : array of shape (B, N)
        Non-negative parameters of the lower factors, N = n(n-1)/2.
    upper_t : array of shape (B, N)
        Non-negative parameters of the upper factors.
    diag : array of shape (B, n), optional
        Positive diagonal entries of D.  Defaults to the identity.

    Returns
    -------
    array of shape (B, n, n)
        ``A[b] = L(lower_t[b]) @ D[b] @ U(upper_t[b])``.

    Implementation: start from the identity and multiply on the right by one
    elementary factor at a time.  Right-multiplying by a bidiagonal factor is
    a single column operation, which is cheaper than a full matrix product:

    * ``M @ L_k(t)``: column k  += t * column k+1
    * ``M @ U_k(t)``: column k+1 += t * column k
    * ``M @ D``     : column j  *= d_j
    """
    lower_t = np.asarray(lower_t, dtype=float)
    upper_t = np.asarray(upper_t, dtype=float)
    batch, count = lower_t.shape
    # Recover n from N = n(n-1)/2.
    n = int(round((1 + np.sqrt(1 + 8 * count)) / 2))
    if num_params(n) != count or upper_t.shape != (batch, count):
        raise ValueError("parameter arrays have inconsistent shapes")

    mat = np.broadcast_to(np.eye(n), (batch, n, n)).copy()

    # Lower-unitriangular part.
    for idx, k in enumerate(lower_word(n)):
        mat[:, :, k] += lower_t[:, idx, None] * mat[:, :, k + 1]

    # Diagonal part (scales columns).
    if diag is not None:
        mat *= np.asarray(diag, dtype=float)[:, None, :]

    # Upper-unitriangular part.
    for idx, k in enumerate(upper_word(n)):
        mat[:, :, k + 1] += upper_t[:, idx, None] * mat[:, :, k]

    return mat


def build_tnn_exact(lower_t: Sequence, upper_t: Sequence,
                    diag: Optional[Sequence] = None) -> List[List[Fraction]]:
    """Exact (Fraction) version of :func:`build_tnn_batch` for one matrix.

    Floats are converted with ``Fraction(x)``, which is exact, so the matrix
    returned is exactly the product of the factors whose parameters are the
    given binary floating-point numbers.  This is what makes the exact
    re-check of a candidate counterexample a rigorous certificate.
    """
    lower = [Fraction(x) for x in lower_t]
    upper = [Fraction(x) for x in upper_t]
    count = len(lower)
    n = int(round((1 + (1 + 8 * count) ** 0.5) / 2))
    if num_params(n) != count or len(upper) != count:
        raise ValueError("parameter lists have inconsistent lengths")

    mat = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]

    # Same column operations as the NumPy version, done exactly.
    for t, k in zip(lower, lower_word(n)):
        for i in range(n):
            mat[i][k] += t * mat[i][k + 1]
    if diag is not None:
        for j, d in enumerate(diag):
            d = Fraction(d)
            for i in range(n):
                mat[i][j] *= d
    for s, k in zip(upper, upper_word(n)):
        for i in range(n):
            mat[i][k + 1] += s * mat[i][k]
    return mat


def j_matrix(mu: Partition) -> np.ndarray:
    """Return J_mu: block diagonal with all-ones blocks of sizes mu_1, mu_2, ...

    Each J_mu is TNN and ``Imm_lambda(J_mu) = mu! * delta_{lambda mu}``
    (Lemma 6.1 of the hooks paper (Lemma 3.1 of the n <= 10 paper)).
    """
    n = sum(mu)
    mat = np.zeros((n, n))
    start = 0
    for part in mu:
        mat[start:start + part, start:start + part] = 1.0
        start += part
    return mat


def min_minor(mat: np.ndarray) -> float:
    """Return the smallest minor of a square matrix (all orders, all positions).

    A matrix is TNN iff this is >= 0 and TP iff it is > 0.  There are
    ``C(2n, n) - 1`` minors, which is fine for the small n used here (3431
    minors for n = 7).  Used only in the test-suite.
    """
    n = mat.shape[0]
    smallest = np.inf
    for size in range(1, n + 1):
        for rows in combinations(range(n), size):
            sub_rows = mat[list(rows)]
            for cols in combinations(range(n), size):
                value = np.linalg.det(sub_rows[:, list(cols)])
                smallest = min(smallest, value)
    return float(smallest)
