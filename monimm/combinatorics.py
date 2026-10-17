"""Exact symmetric-group combinatorics needed for monomial immanants.

Everything here is computed exactly with integers / ``fractions.Fraction`` and
from first principles (no character tables are needed).

Notation follows Stembridge, "Some conjectures for immanants" (Canad. J. Math. 44 (1992)), and the
papers in paper/ (where psi^mu is written eta^mu):

* A partition of n is a weakly decreasing tuple of positive integers, e.g.
  ``(3, 2, 1)``.  Partitions also index conjugacy classes of S_n (cycle types).
* For a cycle type rho, ``z(rho) = prod_i i^{m_i} m_i!`` where ``m_i`` is the
  number of parts equal to i.  The class of type rho has ``n!/z(rho)`` elements.
* For a partition mu, ``psi^mu = Ind_{S_mu}^{S_n} 1`` is the permutation
  character of S_n acting on cosets of the Young subgroup S_mu.
* The class functions ``g_lambda`` are the dual basis to ``{psi^mu}`` under the
  inner product ``<phi, psi> = (1/n!) sum_sigma phi(sigma) psi(sigma)``, i.e.
  ``<g_lambda, psi^mu> = delta_{lambda mu}``.
  The monomial immanant is ``Imm_lambda = d_{g_lambda}``.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import factorial, prod
from typing import Dict, List, Sequence, Tuple

Partition = Tuple[int, ...]


# ---------------------------------------------------------------------------
# Partitions and cycle types
# ---------------------------------------------------------------------------

def partitions(n: int) -> List[Partition]:
    """Return all partitions of ``n`` in reverse lexicographic order.

    This is the order used throughout, e.g. for n = 4:
    ``(4), (3,1), (2,2), (2,1,1), (1,1,1,1)``.  The first partition ``(n)``
    corresponds to the permanent side, the last ``(1^n)`` to the determinant.
    """
    result: List[Partition] = []

    def build(remaining: int, max_part: int, prefix: List[int]) -> None:
        # Once nothing is left, ``prefix`` is a complete partition.
        if remaining == 0:
            result.append(tuple(prefix))
            return
        # Try the largest admissible next part first, giving reverse-lex order.
        for part in range(min(remaining, max_part), 0, -1):
            prefix.append(part)
            build(remaining - part, part, prefix)
            prefix.pop()

    build(n, n, [])
    return result


def cycle_type(perm: Sequence[int]) -> Partition:
    """Return the cycle type of a permutation given in one-line notation.

    ``perm[i]`` is the image of ``i`` (0-indexed).  The result is the
    partition formed by the cycle lengths, sorted in decreasing order.
    """
    n = len(perm)
    seen = [False] * n
    lengths = []
    for start in range(n):
        if seen[start]:
            continue
        # Walk the cycle containing ``start`` and count its length.
        length = 0
        j = start
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            length += 1
        lengths.append(length)
    return tuple(sorted(lengths, reverse=True))


def z_value(rho: Partition) -> int:
    """Return ``z(rho) = prod_i i^{m_i} * m_i!`` (centralizer order).

    The conjugacy class of cycle type ``rho`` in S_n has ``n!/z(rho)``
    elements, so ``(1/n!) * sum over the class = 1/z(rho)`` per class.
    """
    mult: Dict[int, int] = {}
    for part in rho:
        mult[part] = mult.get(part, 0) + 1
    return prod(i ** m * factorial(m) for i, m in mult.items())


def partition_factorial(mu: Partition) -> int:
    """Return ``mu! = mu_1! mu_2! ...``, the order of the Young subgroup S_mu."""
    return prod(factorial(part) for part in mu)


# ---------------------------------------------------------------------------
# Permutation characters psi^mu and the dual basis g_lambda
# ---------------------------------------------------------------------------

def psi_value(mu: Partition, rho: Partition) -> int:
    """Return ``psi^mu(sigma)`` for any sigma of cycle type ``rho``.

    ``psi^mu(sigma)`` is the number of cosets of S_mu fixed by sigma, which
    equals the number of ordered set partitions of {1..n} into blocks of sizes
    ``mu_1, mu_2, ...`` that are unions of cycles of sigma.  Equivalently, it is
    the number of ways to assign each cycle of sigma to a block so that the
    cycle lengths assigned to block j add up to ``mu_j``.

    (In symmetric-function language this is ``<p_rho, h_mu>``.)
    """
    cycles = tuple(rho)

    @lru_cache(maxsize=None)
    def count(index: int, capacities: Tuple[int, ...]) -> int:
        # All cycles placed: valid only if every block is exactly filled.
        if index == len(cycles):
            return 1 if all(c == 0 for c in capacities) else 0
        length = cycles[index]
        total = 0
        # Put cycle ``index`` into any block with enough remaining room.
        for j, cap in enumerate(capacities):
            if cap >= length:
                new_caps = capacities[:j] + (cap - length,) + capacities[j + 1:]
                total += count(index + 1, new_caps)
        return total

    return count(0, tuple(mu))


def _invert_fraction_matrix(matrix: List[List[Fraction]]) -> List[List[Fraction]]:
    """Invert a square matrix of Fractions exactly by Gauss-Jordan elimination."""
    size = len(matrix)
    # Augment [M | I] and row-reduce the left half to the identity.
    aug = [list(row) + [Fraction(int(i == j)) for j in range(size)]
           for i, row in enumerate(matrix)]
    for col in range(size):
        pivot = next(r for r in range(col, size) if aug[r][col] != 0)
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_val = aug[col][col]
        aug[col] = [x / pivot_val for x in aug[col]]
        for r in range(size):
            if r != col and aug[r][col] != 0:
                factor = aug[r][col]
                aug[r] = [a - factor * b for a, b in zip(aug[r], aug[col])]
    return [row[size:] for row in aug]


@lru_cache(maxsize=None)
def monomial_class_functions(n: int) -> Tuple[Tuple[Partition, ...], Tuple[Tuple[Fraction, ...], ...]]:
    """Return ``(parts, G)`` with ``G[a][b] = g_{parts[a]}(class parts[b])``.

    Rows are indexed by lambda (the monomial immanant), columns by the cycle
    type rho at which ``g_lambda`` is evaluated.  Both use the order of
    :func:`partitions`.

    Derivation.  For class functions,
        <g, psi^mu> = sum_rho g(rho) * psi^mu(rho) / z(rho).
    Put ``M[rho][mu] = psi^mu(rho) / z(rho)``.  The duality conditions
    ``<g_lambda, psi^mu> = delta_{lambda mu}`` say ``G M = I``, so ``G = M^{-1}``.
    ``M`` is invertible because {psi^mu} is a basis of the class functions
    (Kostka unitriangularity / Young's rule).
    """
    parts = tuple(partitions(n))
    m_matrix = [[Fraction(psi_value(mu, rho), z_value(rho)) for mu in parts]
                for rho in parts]
    g_matrix = _invert_fraction_matrix(m_matrix)
    return parts, tuple(tuple(row) for row in g_matrix)


# ---------------------------------------------------------------------------
# Classification of shapes already covered by published theorems
# ---------------------------------------------------------------------------

def known_status(mu: Partition) -> str:
    """Return why ``Imm_mu >= 0`` on TNN_n is already known, or ``""`` if open.

    These are the shapes covered by published theorems on TNN_n, independently of the
    Kazhdan-Lusztig computation:

    * hooks ``(a, 1^k)``                         -- Dong-Wu-Gao
    * at most two columns (``mu_1 <= 2``)         -- Clearman-Shelton-Skandera
    * rectangles ``(a^k)``                       -- Stembridge
    * at most two rows and ``n <= 5``             -- Stembridge (computation)

    Shapes outside this list (e.g. ``(4,2)`` and ``(3,2,1)`` for n = 6) are the ones
    that no such theorem covers; for n <= 10 they follow from the computation
    (:data:`monimm.ordering.MN_PROVED_NMAX`).
    """
    n = sum(mu)
    if len(mu) == 1 or mu[1] == 1:
        return "hook (Dong-Wu-Gao)"
    if mu[0] <= 2:
        return "two columns (Clearman-Shelton-Skandera)"
    if len(set(mu)) == 1:
        return "rectangle (Stembridge)"
    if len(mu) <= 2 and n <= 5:
        return "two rows, n<=5 (Stembridge)"
    return ""
