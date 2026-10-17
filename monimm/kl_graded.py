"""Haiman's q-conjecture: the polynomials phi^lambda_q(q^{l(w)/2} C'_w) on extremal pairs.

The statement
-------------
Let H_n(q) be the Hecke algebra of S_n, with (T_s - q)(T_s + 1) = 0, and
C'_w = q^{-l(w)/2} sum_{v <= w} P_{v,w}(q) T_v its Kazhdan-Lusztig basis.  Let
eta^mu_q be the character of H_n(q) induced from the index representation
(T_s -> q) of the parabolic subalgebra H_mu, and define the q-monomial characters
phi^lambda_q by the same inverse-Kostka relation as at q = 1, so that their
Frobenius characteristic is m_lambda.  Haiman (J. AMS 6 (1993), Conj. 2.1) conjectured

    c_{lambda,w}(q) := phi^lambda_q(q^{l(w)/2} C'_w) = sum_{v <= w} P_{v,w}(q) phi^lambda_q(T_v)

lies in N[q], and is symmetric and unimodal about q^{l(w)/2}.  He reports a
computer check for n <= 7.  At q = 1 this is the c_{lambda,w} of
:mod:`monimm.kl_extremal`.

Traces of T_v: Geck-Pfeiffer class polynomials
----------------------------------------------
phi^lambda_q(T_v) is not constant on conjugacy classes, so it is needed for every
v.  We use the class polynomials of Geck and Pfeiffer (Characters of finite
Coxeter groups and Iwahori-Hecke algebras, 2000, Ch. 8): for every trace tau,

  * if u -> s u s is a cyclic shift with l(sus) = l(u), then tau(T_u) = tau(T_{sus});
  * if l(sus) = l(u) - 2, then T_u = T_s T_{sus} T_s and
        tau(T_u) = tau(T_s^2 T_{sus}) = (q - 1) tau(T_{su}) + q tau(T_{sus});
  * every element reaches a minimal-length element of its class by cyclic shifts that
    never increase length (GP Thm 3.2.9), and tau(T_c) is the same for all
    minimal-length c in a class (GP Thm 8.2.3).

So T_v = sum_rho f_{v,rho}(q) T_{c_rho} modulo [H, H], with integer polynomials
f_{v,rho}.  We compute them by increasing length: the elements reachable from v by
length-preserving shifts all have the same f; if one of them, u, has a simple s with
l(sus) = l(u) - 2 we apply the second rule (su and sus are shorter, hence already
done), and otherwise v has minimal length and f_v is the unit vector of its class.

At the minimal-length element c_rho (a product of Coxeter elements of consecutive
blocks of sizes rho_1, rho_2, ...) the irreducible characters are
chi^{(k-j,1^j)}_q(T_{c_(k)}) = (-1)^j q^{k-1-j} for one block (Starkey; Ram 1991),
and for several blocks the graded Frobenius characteristic is the product

    ch(T_{c_rho}) = prod_k sum_j (-1)^j q^{rho_k - 1 - j} s_{(rho_k - j, 1^j)}.

Expanding in the h basis gives phi^lambda_q(T_{c_rho}) as the coefficient of h_lambda,
because p_rho = sum_lambda phi^lambda(rho) h_lambda.

Coefficients
------------
As at q = 1 (see :mod:`monimm.kl_extremal`), P_{v,w} is constant on the double cosets
W_I x W_J of I = D_L(w), J = D_R(w), so

    c_{lambda,w}(q) = sum_{x extremal for w} P_{x,w}(q) G_lambda(I, J, x)(q),
    G_lambda(I, J, x)(q) = sum_{v in W_I x W_J} phi^lambda_q(T_v),

with the same depth-first run and table hash as :func:`monimm.kl_extremal._c_by_class`;
only the accumulated quantity is now a vector of polynomials instead of a cycle-type
count.  deg phi^lambda_q(T_v) <= l(v) and deg P_{v,w} <= (l(w) - l(v) - 1)/2, so
deg c_{lambda,w} <= l(w); a coefficient beyond the stored degree raises an error.

Checks
------
``tests/test_kl_graded.py`` compares, for n <= 4 (n <= 5 was also run once) and q = 2, 3, every c_{lambda,w}(q) with
the trace of sum_v P_{v,w}(q) T_v on the induced modules M^mu = H (x) _{H_mu} index,
built directly from the Hecke relations (no class polynomials, no character formula),
and checks phi^lambda_q(T_v) at q = 1 against phi^lambda(v).  The driver
``run_kl_graded.py`` checks c_{lambda,w}(1) against :mod:`monimm.kl_extremal`, the
symmetries w -> w^{-1}, w_0 w w_0 and stability, then positivity, symmetry and
unimodality.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from fractions import Fraction
from typing import Dict, List, Tuple

import numpy as np
from numba import njit, prange

from .combinatorics import monomial_class_functions, partitions, z_value
from .ordering import character
from .kl_extremal import (ExtremalKL, _block_ids, _sizes, _table_hash, _bsearch)


# ---------------------------------------------------------------------------
# phi^lambda_q at the minimal-length class representatives
# ---------------------------------------------------------------------------

def _padd(a: Dict[int, Fraction], b: Dict[int, Fraction], s: Fraction = Fraction(1)):
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + s * v
    return {k: v for k, v in out.items() if v}


def _pmul(a, b):
    out: Dict[int, Fraction] = defaultdict(Fraction)
    for i, x in a.items():
        for j, y in b.items():
            out[i + j] += x * y
    return {k: v for k, v in out.items() if v}


def minimal_class_values(n: int) -> np.ndarray:
    """B[rho, lambda, k] = coefficient of q^k in phi^lambda_q(T_{c_rho}).

    Rows and columns are indexed by ``partitions(n)`` (the order of BigTables.parts)."""
    parts = [tuple(p) for p in partitions(n)]
    pidx = {p: i for i, p in enumerate(parts)}
    _, g = monomial_class_functions(n)          # g[lambda][rho] = phi^lambda(rho)

    def schur_p(shape):
        k = sum(shape)
        return {tuple(r): {0: Fraction(character(shape, tuple(r)), z_value(tuple(r)))}
                for r in map(tuple, partitions(k))}

    def sf_mul(A, B):
        out = defaultdict(dict)
        for ra, pa in A.items():
            for rb, pb in B.items():
                rr = tuple(sorted(ra + rb, reverse=True))
                out[rr] = _padd(out[rr], _pmul(pa, pb))
        return {r: p for r, p in out.items() if p}

    def ch_coxeter(k):
        out = {}
        for j in range(k):
            sh = tuple([k - j] + [1] * j)
            for r, pl in schur_p(sh).items():
                out[r] = _padd(out.get(r, {}), _pmul(pl, {k - 1 - j: Fraction((-1) ** j)}))
        return out

    maxlen = n * (n - 1) // 2
    B = np.zeros((len(parts), len(parts), maxlen + 1), dtype=np.int64)
    cox = {}
    for rho in parts:
        A = {(): {0: Fraction(1)}}
        for k in rho:
            if k not in cox:
                cox[k] = ch_coxeter(k)
            A = sf_mul(A, cox[k])
        for lam_i in range(len(parts)):
            poly: Dict[int, Fraction] = {}
            for r, pl in A.items():
                c = g[lam_i][pidx[r]]
                if c:
                    poly = _padd(poly, pl, Fraction(c))
            for k, v in poly.items():
                if v.denominator != 1:
                    raise ArithmeticError("non-integral Hecke character value")
                B[pidx[rho], lam_i, k] = int(v)
    return B


# ---------------------------------------------------------------------------
# Class polynomials and traces of every T_v
# ---------------------------------------------------------------------------

@njit(cache=True)
def _class_polys(order, length, left, right, ctype, nct, D):
    """F[v, rho, k]: T_v = sum_rho f_{v,rho}(q) T_{c_rho} mod [H, H] (Geck-Pfeiffer)."""
    N = order.shape[0]
    n1 = left.shape[0]
    F = np.zeros((N, nct, D), np.int64)
    done = np.zeros(N, np.bool_)
    stamp = np.full(N, -1, np.int64)
    queue = np.empty(N, np.int64)
    for idx in range(N):
        v = order[idx]
        if done[v]:
            continue
        lv = length[v]
        # breadth-first search over length-preserving cyclic shifts
        head = 0
        tail = 1
        queue[0] = v
        stamp[v] = v
        found_u = -1
        found_s = -1
        while head < tail and found_u < 0:
            u = queue[head]
            head += 1
            for i in range(n1):
                su = left[i, u]
                sus = right[i, su]
                if length[sus] == lv - 2:
                    found_u = u
                    found_s = i
                    break
                if length[sus] == lv and stamp[sus] != v:
                    stamp[sus] = v
                    queue[tail] = sus
                    tail += 1
        if found_u < 0:
            # v has minimal length in its class (and so does its whole component)
            for t in range(tail):
                F[queue[t], ctype[v], 0] = 1
        else:
            su = left[found_s, found_u]
            sus = right[found_s, su]
            row = np.zeros((nct, D), np.int64)
            for r in range(nct):
                for k in range(D):
                    a = F[su, r, k]
                    if a != 0:
                        row[r, k] -= a                 # (q - 1) f_{su}
                        if k + 1 >= D:
                            row[0, 0] = 1 << 62        # poison: degree overflow
                        else:
                            row[r, k + 1] += a
                    b = F[sus, r, k]
                    if b != 0:                          # q f_{sus}
                        if k + 1 >= D:
                            row[0, 0] = 1 << 62
                        else:
                            row[r, k + 1] += b
            # the component is not fully explored when we stop early, but all of it
            # that we did reach shares f_v; the rest is reached again from its own start
            for t in range(tail):
                F[queue[t]] = row
        for t in range(tail):
            done[queue[t]] = True
    return F


@njit(parallel=True, cache=True)
def _traces(F, B, D):
    """Tr[v, lambda, k] = coefficient of q^k in phi^lambda_q(T_v) = sum_rho f_{v,rho} B[rho, lambda]."""
    N, nct, _ = F.shape
    nl = B.shape[1]
    Tr = np.zeros((N, nl, D), np.int64)
    err = np.zeros(N, np.int64)
    for v in prange(N):
        for r in range(nct):
            for a in range(D):
                f = F[v, r, a]
                if f == 0:
                    continue
                for l in range(nl):
                    for b in range(D):
                        x = B[r, l, b]
                        if x != 0:
                            if a + b >= D:
                                err[v] = 1
                            else:
                                Tr[v, l, a + b] += f * x
    return Tr, err


def hecke_traces(K: ExtremalKL) -> np.ndarray:
    """Tr[v, lambda, k] for every v in S_n (BigTables order) and every lambda |- n."""
    T = K.tables
    n = K.n
    D = n * (n - 1) // 2 + 1
    nct = len(T.parts)
    F = _class_polys(T.order, T.length, T.left, T.right, T.ctype.astype(np.int64), nct, D)
    if np.abs(F).max() >= 1 << 61:
        raise ArithmeticError("class polynomial degree overflow")
    B = minimal_class_values(n)
    Tr, err = _traces(F, B, D)
    if err.any():
        raise ArithmeticError("trace degree overflow")
    return Tr


# ---------------------------------------------------------------------------
# c_{lambda,w}(q)
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _cq_by_class(class_ws, class_start, perms, Ld, Rd, start, size, xs, pid, polys,
                 Tr, H, logfact, out, err):
    """As kl_extremal._c_by_class, accumulating the polynomials Tr[v] over each double coset."""
    ncls = class_start.shape[0] - 1
    n = perms.shape[1]
    nl = Tr.shape[1]
    D = Tr.shape[2]
    Dp = polys.shape[1]
    for g in prange(ncls):
        lo = class_start[g]
        hi = class_start[g + 1]
        w0 = class_ws[lo]
        L = Ld[w0]
        R = Rd[w0]
        pb = np.empty(n, np.int64)
        vb = np.empty(n, np.int64)
        r = _block_ids(R, n, pb)
        c = _block_ids(L, n, vb)
        tot = 0
        for k in range(lo, hi):
            tot += size[class_ws[k]]
        keys = np.empty(tot, np.int64)
        i = 0
        for k in range(lo, hi):
            w = class_ws[k]
            for t in range(start[w], start[w] + size[w]):
                keys[i] = xs[t]
                i += 1
        keys = np.unique(keys)
        nk = keys.shape[0]
        hmask = 1
        while hmask + 1 < 2 * nk:
            hmask = hmask * 2 + 1
        hkey = np.zeros(hmask + 1, np.uint64)
        hidx = np.full(hmask + 1, -1, np.int64)
        expect = np.zeros(nk, np.int64)
        alpha = np.zeros(r, np.int64)
        beta = np.zeros(c, np.int64)
        _sizes(pb, r, alpha)
        _sizes(vb, c, beta)
        base = 0.0
        for a in range(r):
            base += logfact[alpha[a]]
        for b in range(c):
            base += logfact[beta[b]]
        T = np.zeros((r, c), np.int64)
        for j in range(nk):
            p = perms[keys[j]]
            h = _table_hash(p, pb, vb, H)
            slot = np.int64(h & np.uint64(hmask))
            while hidx[slot] != -1:
                if hkey[slot] == h:
                    err[0] = 7
                slot = (slot + 1) & hmask
            hkey[slot] = h
            hidx[slot] = j
            for a in range(r):
                for b in range(c):
                    T[a, b] = 0
            for t in range(n):
                T[pb[t], vb[p[t]]] += 1
            lg = base
            for a in range(r):
                for b in range(c):
                    lg -= logfact[T[a, b]]
            expect[j] = np.int64(np.round(np.exp(lg)))
        acc = np.zeros((nk, nl, D), np.int64)
        hits = np.zeros(nk, np.int64)
        used = np.zeros(n, np.bool_)
        choice = np.full(n, -1, np.int64)
        part = np.zeros(n + 1, np.uint64)
        depth = 0
        rank = 0
        while depth >= 0:
            if choice[depth] >= 0:
                used[choice[depth]] = False
            v = choice[depth] + 1
            while v < n and used[v]:
                v += 1
            if v == n:
                choice[depth] = -1
                depth -= 1
                continue
            choice[depth] = v
            used[v] = True
            part[depth + 1] = part[depth] + H[pb[depth], vb[v]]
            if depth == n - 1:
                h = part[n]
                slot = np.int64(h & np.uint64(hmask))
                while hidx[slot] != -1:
                    if hkey[slot] == h:
                        j = hidx[slot]
                        hits[j] += 1
                        for l in range(nl):
                            for k in range(D):
                                acc[j, l, k] += Tr[rank, l, k]
                        break
                    slot = (slot + 1) & hmask
                rank += 1
            else:
                depth += 1
        for j in range(nk):
            if hits[j] != expect[j]:
                err[0] = 6
        for k in range(lo, hi):
            w = class_ws[k]
            for t in range(start[w], start[w] + size[w]):
                j = _bsearch(keys, 0, nk, xs[t])
                P = polys[pid[t]]
                for a in range(Dp):
                    pa = P[a]
                    if pa == 0:
                        continue
                    for l in range(nl):
                        for b in range(D):
                            x = acc[j, l, b]
                            if x != 0:
                                if a + b >= D:
                                    err[0] = 8
                                else:
                                    out[l, w, a + b] += pa * x


def c_polynomials(K: ExtremalKL, Tr: np.ndarray = None) -> np.ndarray:
    """C[lambda, w, k] = coefficient of q^k in c_{lambda,w}(q) = phi^lambda_q(q^{l(w)/2} C'_w)."""
    T = K.tables
    if Tr is None:
        Tr = hecke_traces(K)
    n = K.n
    t = time.time()
    key = K.Ld.astype(np.int64) << 32 | K.Rd.astype(np.int64)
    order = np.argsort(key, kind="stable")
    sk = key[order]
    cuts = np.flatnonzero(np.diff(sk)) + 1
    class_start = np.concatenate([[0], cuts, [len(sk)]]).astype(np.int64)
    out = np.zeros((Tr.shape[1], T.N, Tr.shape[2]), dtype=np.int64)
    H = np.random.default_rng(12345).integers(0, 2 ** 63, size=(n, n), dtype=np.uint64)
    H = H * np.uint64(2) + np.uint64(1)
    logfact = np.array([math.lgamma(k + 1) for k in range(n + 1)])
    err = np.zeros(1, dtype=np.int64)
    _cq_by_class(order.astype(np.int64), class_start, T.perms, K.Ld, K.Rd, K.start, K.size,
                 K.xs, K.pid, K.polys.astype(np.int64), Tr, H, logfact, out, err)
    if err[0] == 6:
        raise ArithmeticError("double coset hit count differs from its size")
    if err[0] == 7:
        raise ArithmeticError("two needed double cosets share a hash; change the seed")
    if err[0] == 8:
        raise ArithmeticError("deg c_{lambda,w} exceeds l(w)")
    K.timings["q-coefficients"] = time.time() - t
    return out


# ---------------------------------------------------------------------------
# Properties of the polynomials
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _properties(C, length, rows):
    """Per w: [negative coefficients, asymmetric rows, non-unimodal rows, degree > l(w)],
    over the lambda in ``rows``.  Unimodal = the non-zero steps of the coefficient
    sequence between its first and last non-zero entry never go down and then up."""
    N = C.shape[1]
    D = C.shape[2]
    res = np.zeros((N, 4), np.int64)
    bad_l = np.full(N, -1, np.int64)            # first non-unimodal lambda at w
    for w in prange(N):
        lw = length[w]
        for li in range(rows.shape[0]):
            l = rows[li]
            for k in range(D):
                if C[l, w, k] < 0:
                    res[w, 0] += 1
                if k > lw and C[l, w, k] != 0:
                    res[w, 3] += 1
            for k in range(lw + 1):
                if C[l, w, k] != C[l, w, lw - k]:
                    res[w, 1] += 1
                    break
            lo = 0
            while lo <= lw and C[l, w, lo] == 0:
                lo += 1
            hi = lw
            while hi >= lo and C[l, w, hi] == 0:
                hi -= 1
            went_down = False
            for k in range(lo, hi):
                d = C[l, w, k + 1] - C[l, w, k]
                if d < 0:
                    went_down = True
                elif d > 0 and went_down:
                    res[w, 2] += 1
                    if bad_l[w] < 0:
                        bad_l[w] = l
                    break
    return res, bad_l


def check_properties(C: np.ndarray, length: np.ndarray, rows=None) -> Dict[str, object]:
    """Non-negativity, symmetry about q^{l(w)/2}, unimodality, for every (lambda, w).

    ``rows`` restricts to a subset of the lambda (default: all), without copying C."""
    rows = np.arange(C.shape[0]) if rows is None else np.asarray(rows, dtype=np.int64)
    res, bad_l = _properties(C, length.astype(np.int64), rows)
    if res[:, 3].any():
        raise ArithmeticError("degree above l(w)")
    ws = np.flatnonzero(res[:, 2])[:5]
    examples: List[Tuple[int, int]] = [(int(bad_l[w]), int(w)) for w in ws]
    return {"negative_coefficients": int(res[:, 0].sum()), "asymmetric": int(res[:, 1].sum()),
            "non_unimodal": int(res[:, 2].sum()), "non_unimodal_examples": examples}


def accumulation_bound(K: ExtremalKL, Tr: np.ndarray) -> int:
    """An upper bound for every partial sum in :func:`c_polynomials`.

    |c_{lambda,w}| coefficientwise is at most (sum of P_{x,w} coefficients) * (sum over the
    double coset of the coefficient sums of |phi_q(T_v)|), and the double cosets of the
    extremal x of w are disjoint, so every partial sum is bounded by
    max P(1) * n! * max_v sum_k |Tr[v, lambda, k]|.  Below 2^63 means no int64 wraparound."""
    m = 0
    for l in range(Tr.shape[1]):
        m = max(m, int(np.abs(Tr[:, l, :]).sum(axis=1).max()))
    return int(K.values_at_one.max()) * K.tables.N * m
