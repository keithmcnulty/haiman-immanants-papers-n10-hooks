"""Shareshian-Wachs chromatic quasisymmetric functions of unit interval graphs: an independent check of
:mod:`monimm.kl_graded` at codominant w.

Statement used
--------------
For w codominant (312-avoiding) let m(i) = max(w(1), ..., w(i)) (a Hessenberg function) and G_w the graph on [n]
with edges {i < j : j <= m(i)}.  Clearman-Hyatt-Shelton-Skandera (EJC 23(2) (2016) P2.7, Cor. 7.5, with the
labelling of their (7.3)) give

    X_{G_w}(x, q) = sum_lam phi^lam_q(q^{l(w)/2} C'_w) e_lam,

where X_G(x, q) = sum over proper colourings kappa of q^{asc(kappa)} x^kappa and asc counts edges i < j with
kappa(i) < kappa(j) (Shareshian-Wachs).  Nothing here uses the Hecke algebra, so comparing with
monimm.kl_graded.c_polynomials checks the whole graded layer (class polynomials, traces, coset sums) on every
codominant w.  The comparison is check_codominant_chromatic.py (results/codominant_chromatic_n{8,9}.json)
and, for n = 10, run_kl_graded.py --codominant (results/kl_graded_n10_codominant.json).

Computation
-----------
The coefficient of m_mu in X_G is the sum of q^asc over colourings using colour t exactly mu_t times
(t = 1, 2, ...).  Colour classes are independent sets of G.  We add them in colour order: a state is the set S of
coloured vertices, and adding a class C (colour larger than all used) creates one ascent for every edge (i, j)
with i < j, i in S, j in C.  Then e_lam = sum_mu N[lam, mu] m_mu (N = number of 0/1 matrices with row sums lam and
column sums mu), so the e-coefficients are the m-coefficients times N^{-1} (exact, integer).
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np
from numba import njit, prange

from .combinatorics import partitions


def codominant_perms(n: int) -> List[Tuple[int, ...]]:
    """All 312-avoiding permutations of {0..n-1} (one-line), generated recursively (Catalan many)."""
    out = []

    def rec(prefix, remaining):
        if not remaining:
            out.append(tuple(prefix)); return
        for x in sorted(remaining):
            # appending x creates a 312 (c, a, b with a < b < c, positions in that order) iff some earlier c > x has
            # a later a < x between them; check against the prefix
            ok = True
            for i, c in enumerate(prefix):
                if c > x and any(a < x for a in prefix[i + 1:]):
                    ok = False; break
            if ok:
                rec(prefix + [x], remaining - {x})
    rec([], set(range(n)))
    return out


def lower_adjacency(w: Sequence[int]) -> np.ndarray:
    """adj[j] = bitmask of i < j adjacent to j in G_w."""
    n = len(w)
    m = [max(w[:i + 1]) for i in range(n)]
    adj = np.zeros(n, np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            if j <= m[i]:
                adj[j] |= 1 << i
    return adj


def zero_one_matrix(parts: List[Tuple[int, ...]]) -> np.ndarray:
    """N[lam, mu] = number of 0/1 matrices with row sums lam and column sums mu."""
    def count(lam, mu):
        @lru_cache(None)
        def rec(i, cols):
            if i == len(lam):
                return int(all(c == 0 for c in cols))
            tot = 0
            for S in itertools.combinations(range(len(cols)), lam[i]):
                if all(cols[j] > 0 for j in S):
                    nc = list(cols)
                    for j in S: nc[j] -= 1
                    tot += rec(i + 1, tuple(nc))
            return tot
        return rec(0, tuple(mu))
    return np.array([[count(l, m) for m in parts] for l in parts], dtype=object)


def exact_inverse(M: np.ndarray) -> np.ndarray:
    """Inverse of an integer matrix over Q (Gauss-Jordan with Fractions); must come out integral."""
    k = M.shape[0]
    A = [[Fraction(int(M[i, j])) for j in range(k)] + [Fraction(int(i == j)) for j in range(k)] for i in range(k)]
    for c in range(k):
        p = next(r for r in range(c, k) if A[r][c] != 0)
        A[c], A[p] = A[p], A[c]
        piv = A[c][c]
        A[c] = [x / piv for x in A[c]]
        for r in range(k):
            if r != c and A[r][c] != 0:
                f = A[r][c]
                A[r] = [x - f * y for x, y in zip(A[r], A[c])]
    inv = np.array([[A[i][k + j] for j in range(k)] for i in range(k)], dtype=object)
    if any(x.denominator != 1 for x in inv.flat):
        raise ArithmeticError("N^{-1} is not integral")
    return np.array([[int(x) for x in row] for row in inv], dtype=np.int64)


@njit(cache=True)
def _popcount(x):
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


@njit(cache=True)
def _m_coefficients(adj, n, mus, mulen, D):
    """out[p, k] = coefficient of q^k in [m_{mu_p}] X_G (mus[p, :mulen[p]] = the parts, any order)."""
    N = 1 << n
    indep = np.zeros(N, np.bool_)
    for S in range(N):
        ok = True
        for j in range(n):
            if (S >> j) & 1 and (adj[j] & S) != 0:
                ok = False; break
        indep[S] = ok
    size = np.zeros(N, np.int64)
    for S in range(N): size[S] = _popcount(S)
    P = mus.shape[0]
    out = np.zeros((P, D), np.int64)
    F = np.zeros((N, D), np.int64)
    G = np.zeros((N, D), np.int64)
    for p in range(P):
        F[:, :] = 0
        F[0, 0] = 1
        for t in range(mulen[p]):
            k = mus[p, t]
            G[:, :] = 0
            for S in range(N):
                if F[S].any():
                    comp = (N - 1) & ~S
                    C = comp
                    while C:
                        if size[C] == k and indep[C]:
                            a = 0
                            for j in range(n):
                                if (C >> j) & 1:
                                    a += _popcount(adj[j] & S)
                            for d in range(D - a):
                                G[S | C, d + a] += F[S, d]
                        C = (C - 1) & comp
            F[:, :] = G
        out[p] = F[N - 1]
    return out


@njit(parallel=True, cache=True)
def _m_coefficients_many(adjs, n, mus, mulen, D):
    W = adjs.shape[0]
    out = np.zeros((W, mus.shape[0], D), np.int64)
    for i in prange(W):
        out[i] = _m_coefficients(adjs[i], n, mus, mulen, D)
    return out


def e_coefficients(n: int, perms: Sequence[Sequence[int]], parts: List[Tuple[int, ...]] = None,
                   D: int = None) -> np.ndarray:
    """E[lam, i, k] = coefficient of q^k in [e_lam] X_{G_{perms[i]}}(x, q), lam in ``parts`` order."""
    parts = [tuple(p) for p in (parts if parts is not None else partitions(n))]
    D = D if D is not None else n * (n - 1) // 2 + 1
    mus = np.zeros((len(parts), n), np.int64)
    mulen = np.array([len(p) for p in parts], np.int64)
    for i, p in enumerate(parts):
        mus[i, :len(p)] = p
    adjs = np.array([lower_adjacency(w) for w in perms], np.int64)
    Mco = _m_coefficients_many(adjs, n, mus, mulen, D)        # (W, mu, D)
    Ninv = exact_inverse(zero_one_matrix(parts))               # e = N m  =>  m-coeffs = N^T e-coeffs
    # X = sum_mu x_mu m_mu = sum_lam c_lam e_lam = sum_lam c_lam sum_mu N[lam,mu] m_mu  =>  x = N^T c  =>  c = N^{-T} x
    E = np.einsum('lm,wmd->lwd', Ninv.T, Mco)
    return E
