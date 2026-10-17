"""Kazhdan-Lusztig polynomials, KL immanants, and monomial immanants in the KL basis.

Conventions
-----------
Permutations are tuples in one-line notation, 0-indexed: ``w = (w(0), ..., w(n-1))``.
Products are composition, ``(u v)(i) = u(v(i))``.  So for the simple
transposition s_i = (i, i+1):

* ``s_i w`` (left multiplication) swaps the *values* i and i+1 in one-line notation;
* ``w s_i`` (right multiplication) swaps the *positions* i and i+1.

The length l(w) is the number of inversions.  w_0 = (n-1, ..., 0) is the
longest element, and ``w_0 v`` maps each value j of v to n-1-j.

Bruhat order (tableau criterion): x <= w iff for all i, k
#{j <= i : x(j) >= k} <= #{j <= i : w(j) >= k}.

KL polynomials P_{x,w}(q) are computed with the standard recursion
(Kazhdan-Lusztig 1979; Humphreys, *Reflection Groups and Coxeter Groups*, 7.11).
Pick s with s w < w and put v = s w.  Then for x <= w

    P_{x,w} = q^{1-c} P_{sx,v} + q^c P_{x,v}
              - sum_{z : s z < z, z < v} mu(z, v) q^{(l(w)-l(z))/2} P_{x,z},

where c = 1 if s x < x and c = 0 otherwise.  P_{x,y} = 0 unless x <= y, and
mu(z, v) is the coefficient of q^{(l(v)-l(z)-1)/2} in P_{z,v}.

KL immanants (Rhoades-Skandera, "Kazhdan-Lusztig immanants and products of
matrix minors", J. Algebra 304 (2006)):

    Imm_w(x) = sum_{v >= w} (-1)^{l(v)-l(w)} P_{w_0 v, w_0 w}(1) x_{1,v(1)} ... x_{n,v(n)}.

Imm_e = det and Imm_{w_0} = x_{1,n} x_{2,n-1} ... x_{n,1}.  They form a basis
of the n!-dimensional span of the permutation monomials, and for any function
f on S_n

    sum_v f(v) x_{1,v(1)} ... x_{n,v(n)} = sum_w f(C'_w(1)) Imm_w(x),
    C'_w(1) = sum_{v <= w} P_{v,w}(1) v,

which follows from the KL inversion formula.  For the monomial immanant
(f = g_lambda) the coefficients are therefore

    c_{lambda, w} = sum_{v <= w} P_{v,w}(1) g_lambda(cycle type of v).

Rhoades-Skandera show that every Imm_w is >= 0 on TNN matrices.  So
c_{lambda, w} >= 0 for all w would prove Imm_lambda >= 0 on TNN_n.  This is the
positivity in Haiman's conjecture, which implies MN_n (Stembridge; see the
paper's reference [9]).

None of the conventions above are taken on trust.  The test-suite checks the
KL inversion formula and known P_{x,w} in S_4, and :func:`kl_coefficients`
checks the expansion identity exactly on all n! monomials, which a wrong
convention would almost certainly break.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import permutations
from typing import Dict, List, Tuple

from .combinatorics import cycle_type, monomial_class_functions

Perm = Tuple[int, ...]
Poly = Tuple[int, ...]          # coefficients of 1, q, q^2, ... (no trailing zeros)


# ---------------------------------------------------------------------------
# Permutations: length, Bruhat order, simple reflections
# ---------------------------------------------------------------------------

def length(w: Perm) -> int:
    """Coxeter length = number of inversions."""
    return sum(w[i] > w[j] for i in range(len(w)) for j in range(i + 1, len(w)))


def left_mult(i: int, w: Perm) -> Perm:
    """s_i w: swap the values i and i+1 in one-line notation."""
    return tuple(i + 1 if x == i else i if x == i + 1 else x for x in w)


def has_left_descent(i: int, w: Perm) -> bool:
    """s_i w < w, i.e. the value i+1 appears before the value i."""
    return w.index(i + 1) < w.index(i)


def longest_times(v: Perm) -> Perm:
    """w_0 v: replace each value j by n-1-j."""
    n = len(v)
    return tuple(n - 1 - x for x in v)


def bruhat_leq(x: Perm, w: Perm) -> bool:
    """x <= w in Bruhat order (tableau criterion)."""
    n = len(x)
    for i in range(n):
        for k in range(n):
            if (sum(1 for j in range(i + 1) if x[j] >= k)
                    > sum(1 for j in range(i + 1) if w[j] >= k)):
                return False
    return True


# ---------------------------------------------------------------------------
# Integer polynomials in q, as coefficient tuples
# ---------------------------------------------------------------------------

def _trim(coeffs: List[int]) -> Poly:
    while coeffs and coeffs[-1] == 0:
        coeffs.pop()
    return tuple(coeffs)


def _add(a: Poly, b: Poly, shift: int = 0, scale: int = 1) -> Poly:
    """a + scale * q^shift * b."""
    out = list(a) + [0] * max(0, len(b) + shift - len(a))
    for k, c in enumerate(b):
        out[k + shift] += scale * c
    return _trim(out)


def poly_at_one(p: Poly) -> int:
    return sum(p)


# ---------------------------------------------------------------------------
# KL polynomials
# ---------------------------------------------------------------------------

class KazhdanLusztig:
    """All KL polynomials P_{x,w} of S_n, with x <= w.

    Attributes
    ----------
    perms : all permutations, sorted by length
    P : dict (x, w) -> coefficient tuple, only for x <= w
    """

    def __init__(self, n: int):
        self.n = n
        self.perms: List[Perm] = sorted(permutations(range(n)), key=length)
        self.length = {w: length(w) for w in self.perms}
        # Bruhat intervals below each w, needed by the recursion and the sums.
        self.below: Dict[Perm, List[Perm]] = {
            w: [x for x in self.perms if self.length[x] <= self.length[w] and bruhat_leq(x, w)]
            for w in self.perms}
        self.P: Dict[Tuple[Perm, Perm], Poly] = {}
        for w in self.perms:              # increasing length: everything needed is known
            self._compute_column(w)

    def p(self, x: Perm, w: Perm) -> Poly:
        """P_{x,w}, zero unless x <= w."""
        return self.P.get((x, w), ())

    def mu(self, z: Perm, v: Perm) -> int:
        """Coefficient of q^{(l(v)-l(z)-1)/2} in P_{z,v} (0 if l(v)-l(z) is even)."""
        diff = self.length[v] - self.length[z]
        if diff <= 0 or diff % 2 == 0:
            return 0
        coeffs = self.p(z, v)
        k = (diff - 1) // 2
        return coeffs[k] if k < len(coeffs) else 0

    def _compute_column(self, w: Perm) -> None:
        """Fill in P_{x,w} for all x <= w, by the recursion in the module docstring."""
        if self.length[w] == 0:
            self.P[(w, w)] = (1,)
            return
        s = next(i for i in range(self.n - 1) if has_left_descent(i, w))
        v = left_mult(s, w)
        # The correction terms: z < v with s z < z and mu(z, v) != 0.
        corrections = [(z, self.mu(z, v)) for z in self.below[v]
                       if z != v and has_left_descent(s, z) and self.mu(z, v)]
        for x in self.below[w]:
            c = 1 if has_left_descent(s, x) else 0
            poly = _add((), self.p(left_mult(s, x), v), shift=1 - c)   # q^{1-c} P_{sx,v}
            poly = _add(poly, self.p(x, v), shift=c)                # + q^c P_{x,v}
            for z, m in corrections:
                shift = (self.length[w] - self.length[z]) // 2
                poly = _add(poly, self.p(x, z), shift=shift, scale=-m)
            self.P[(x, w)] = poly


# ---------------------------------------------------------------------------
# KL immanants and the expansion of monomial immanants
# ---------------------------------------------------------------------------

def kl_immanant_coefficients(kl: KazhdanLusztig, w: Perm) -> Dict[Perm, int]:
    """Imm_w as {v: coefficient of x_{1,v(1)} ... x_{n,v(n)}}."""
    w0w = longest_times(w)
    out = {}
    for v in kl.perms:
        value = poly_at_one(kl.p(longest_times(v), w0w))   # zero unless v >= w
        if value:
            out[v] = (-1) ** (kl.length[v] - kl.length[w]) * value
    return out


def kl_coefficients(n: int, kl: KazhdanLusztig = None
                    ) -> Tuple[KazhdanLusztig, Dict[Tuple[int, ...], Dict[Perm, Fraction]]]:
    """Coefficients c_{lambda,w} with Imm_lambda = sum_w c_{lambda,w} Imm_w, for all lambda |- n.

    Returns ``(kl, coeffs)`` with ``coeffs[lambda][w] = c_{lambda,w}``.  Before
    returning, the identity is checked exactly: for every lambda and every
    permutation v, the coefficient of x_{1,v(1)} ... x_{n,v(n)} on the right-hand
    side must be g_lambda(v).
    """
    kl = kl or KazhdanLusztig(n)
    parts, g = monomial_class_functions(n)
    g_of = {lam: dict(zip(parts, row)) for lam, row in zip(parts, g)}
    ctype = {v: cycle_type(v) for v in kl.perms}

    coeffs = {}
    for lam in parts:
        coeffs[lam] = {w: sum((poly_at_one(kl.p(v, w)) * Fraction(g_of[lam][ctype[v]])
                               for v in kl.below[w]), Fraction(0))
                       for w in kl.perms}

    # Exact check of the expansion on the basis of permutation monomials.
    imm = {w: kl_immanant_coefficients(kl, w) for w in kl.perms}
    for lam in parts:
        total: Dict[Perm, Fraction] = {}
        for w, c in coeffs[lam].items():
            if c:
                for v, a in imm[w].items():
                    total[v] = total.get(v, Fraction(0)) + c * a
        for v in kl.perms:
            assert total.get(v, 0) == g_of[lam][ctype[v]], (lam, v)
    return kl, coeffs
