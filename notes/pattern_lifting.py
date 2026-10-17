#!/usr/bin/env python3
"""Pattern lifting: singular permutations are not TNN-isolated (Theorem 9.6 of the hooks paper), checked exactly.

Usage
-----
    .venv/bin/python notes/pattern_lifting.py --n 8                      # certify every singular w, ~2 min
    .venv/bin/python notes/pattern_lifting.py --n 8 --lemma all          # + Locality Lemma on all occurrences
    .venv/bin/python notes/pattern_lifting.py --n 10 --lemma 100000      # n = 10: ~16 GB, builds the KL table

Writes results/pattern_lifting_n{n}.json.

Mathematics (see Section 9 of the hooks paper)
----------------------------------------------
For F in the span Pi_n of permutation monomials write F = sum_w a_w(F) Imm_w (Rhoades-Skandera KL immanants);
then a_w(F) = f(C'_w(1)) = sum_{sigma <= w} P_{sigma,w}(1) f(sigma), where F = sum_sigma f(sigma) x^sigma.
w is *TNN-isolated* if a_w(F) >= 0 for every F >= 0 on TNN_n.  Skandera (JPAA 2008, Thm 5.5): smooth w are
isolated.  Theorem 9.6 of the hooks paper: singular w (containing 3412 or 4231) are not.

Witnesses at n = 4 (Proposition 9.3 of the hooks paper; AM-GM certificates re-checked here):
    F = Imm_1432 + Imm_3214 - 2 Imm_3412 >= 0,      G = Imm_2143 + Imm_4321 - Imm_4231 >= 0   on TNN_4.

Lifting.  Let w contain p in {3412, 4231} at positions I (|I| = 4) with values T = w(I), and let
w_c = st(w|_{I^c}).  For W in {F, G},
    H(x) = W(x[I, T]) * Imm_{w_c}(x[I^c, T^c])
is >= 0 on TNN_n (submatrices of TNN matrices are TNN, KL immanants are TNN).  If a_w(H) < 0 then w is
not isolated.  This is an exact integer computation: with sigma ranging over {sigma(I) = T},
    g[s] = sum_{sigma : st(sigma|_I) = s} P_{sigma,w}(1) * f_{w_c}(st(sigma|_{I^c}))      (s in S_4),
    A(u) = sum_s g[s] f_u(s)  = coefficient of Imm_w in Imm_u(x[I,T]) Imm_{w_c}(x[I^c,T^c]),
    a_w(H) = sum_s g[s] W(s) = sum_u c_u A(u)            (W = sum_u c_u Imm_u),
where f_u(s) is the coefficient of x^s in Imm_u.  By Billey-Braden (hooks paper, proof of Theorem 3.1),
A(u) = a_y(w) >= 0 for the y in the coset with pattern (u, w_c).

Locality Lemma (conjectured; checked here): if p = st(w|_I) avoids 321 then A(u) = delta_{u,p} for all u.
For p = 3412 this gives a_w(H) = -2 at every occurrence, so every w containing 3412 is not isolated.
For p = 4231 (which contains 321) A(2143) = 2 occurs, and the lifting must choose the occurrence.

Proof for all n (Section 9 of the hooks paper).  Lemma 1: A(p) = 1 and A(u) = 0 for u > p.  Lemma 2:
A(u) + sum_{u' > u} A(u') P_{u,u'}(1) <= P_{sigma(u, w_c), w}(1).  4231: eps*Imm_2143 + 3/(4 eps)*Imm_4321 - Imm_4231 >= 0 on
TNN_4 for all eps > 0 (n4_inequalities), so a_w = eps*A(2143) - 1 < 0.  4231-avoiding w with 3412: by Lemma S (Lemma 9.5 of the hooks paper)
(w t_ij smooth in X_w iff the points inside the (i, j) rectangle decrease) both swaps of any 3412 occurrence are
smooth, so A(1432) = A(3214) = 0 and a_w = -2.  --bound and --lemma-s check these ingredients exactly.

P_{sigma,w}(1) comes from the extremal-pair KL engine (monimm/kl_extremal.py) for every n, so n <= 8 is
also an independent re-check of the FastKL computation in the scratch run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
import os
import sys
import time
from itertools import combinations, permutations
from math import factorial

import numpy as np
from numba import njit, prange

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from monimm.kazhdan_lusztig import KazhdanLusztig, kl_immanant_coefficients  # noqa: E402
from monimm.kl_extremal import ExtremalKL, _bsearch, _canon_rank, _rank      # noqa: E402

P3412 = (2, 3, 0, 1)
P4231 = (3, 1, 2, 0)


def imm_table(m: int) -> np.ndarray:
    """F[u, v] = coefficient of x^v in Imm_u on S_m, lexicographic ranks (exact integers)."""
    perms = list(permutations(range(m)))
    if m == 0:
        return np.ones((1, 1), dtype=np.int64)
    kl = KazhdanLusztig(m)
    rank = {p: r for r, p in enumerate(perms)}
    F = np.zeros((len(perms), len(perms)), dtype=np.int64)
    for u in perms:
        for v, c in kl_immanant_coefficients(kl, u).items():
            F[rank[u], rank[v]] = int(c)
    return F


def witness_vectors(F4):
    """Coefficient vectors (over S_4, lex ranks) of the n = 4 witnesses, and their KL decodings."""
    perms4 = list(permutations(range(4)))
    r = {p: i for i, p in enumerate(perms4)}
    P = lambda s: tuple(int(c) - 1 for c in s)
    combos = {"3412": {P("1432"): 1, P("3214"): 1, P("3412"): -2},
              "4231": {P("2143"): 1, P("4321"): 1, P("4231"): -1}}
    vec = {k: sum(a * F4[r[u]] for u, a in c.items()) for k, c in combos.items()}
    return combos, vec


def sos_certificate(d, chart, combo):
    """Check an AM-GM certificate: every negative monomial m*u is covered by m and m*u^2."""
    ex = d.exact_expansion(chart, combo)
    neg = {a: -c for a, c in ex.items() if c < 0}
    pos = {a: c for a, c in ex.items() if c > 0}
    used = Counter()
    shifts = Counter()
    for a, need in neg.items():
        cands = []
        for b in pos:
            g = tuple(2 * x - y for x, y in zip(a, b))
            if g in pos and b < g:
                cands.append((b, g))
        assert len(cands) >= 1, a
        b, g = cands[0]
        # need * m u <= need/2 * (m + m u^2): consumes need/2 of each outer monomial
        used[b] += Fraction(need, 2)
        used[g] += Fraction(need, 2)
        shifts[tuple(k + 1 for k in range(len(a)) if a[k] != b[k])] += 1
    assert all(used[m] <= pos[m] for m in used)
    return len(ex), len(neg), shifts


def certify_witnesses():
    """Exact AM-GM certificates that F, G >= 0 on TNN_4 (Proposition 9.3 of the hooks paper)."""
    from monimm.tnn_cone import KLChartData, standard_chart, parse_perm
    d = KLChartData(4)
    ch = standard_chart(4)
    out = {}
    for name, combo in (("F", {"1432": 1, "3214": 1, "3412": -2}), ("G", {"2143": 1, "4321": 1, "4231": -1})):
        n_terms, n_neg, _ = sos_certificate(d, ch, {parse_perm(k): v for k, v in combo.items()})
        out[name] = {"chart_terms": n_terms, "negative_terms_covered": n_neg}
    return out


def n4_inequalities():
    """Exact check of the two n = 4 inequalities used in the proof (hooks paper, Proposition 9.3):

        Imm_3412^2 <= Imm_1432 * Imm_3214,      Imm_4231^2 <= 3 * Imm_2143 * Imm_4321      on TNN_4,

    where Imm_4321 = x14 x23 x32 x41 (NOT det: det = 1 in this unitriangular chart, and Imm_4231^2 <= 3 Imm_2143 det is
    false on TP_4, e.g. at the chart point t_1 = ... = t_12 = 10).

    Chart (hooks paper): A = L_3(t1) L_2(t2) L_1(t3) L_3(t4) L_2(t5) L_3(t6) * U_3(t7) U_2(t8) U_1(t9) U_3(t10) U_2(t11) U_3(t12),
    L_i(t) = I + t E_{i+1,i}, U_i(t) = I + t E_{i,i+1}; both words s3 s2 s1 s3 s2 s3 are reduced words of w_0.  Positive
    parameters give every totally positive matrix up to a positive diagonal factor, which multiplies every immanant by
    its determinant, and TP_4 is dense in TNN_4.  In the chart:
      * Imm_1432, Imm_3214, Imm_2143, Imm_4321 have only positive coefficients;
      * Imm_3412 = m is one monomial with m^2 = a b, a a monomial of Imm_1432 and b one of Imm_3214, so
        Imm_3412^2 = a b <= Imm_1432 Imm_3214;
      * Imm_4231 = m1 + m2 + m3 with m1^2 = a1 b1, m2^2 = a2 b1, m3^2 = a2 b2 (a1 != a2 monomials of Imm_2143, b1 != b2
        of Imm_4321), so by Cauchy-Schwarz Imm_4231^2 <= 3 (m1^2 + m2^2 + m3^2) <= 3 (a1 + a2)(b1 + b2) <= 3 Imm_2143 Imm_4321.
    Returns the monomials as strings (for the paper).
    """
    import sympy as sp
    F4 = imm_table(4)
    perms = list(permutations(range(4)))
    r = {p: i for i, p in enumerate(perms)}
    t = sp.symbols("t1:13")
    M = sp.eye(4)
    for k, i in enumerate([3, 2, 1, 3, 2, 3]):           # lower factors, left to right (1-indexed s_i)
        E = sp.eye(4); E[i, i - 1] = t[k]; M = M * E
    for k, i in enumerate([3, 2, 1, 3, 2, 3]):           # upper factors, left to right
        E = sp.eye(4); E[i - 1, i] = t[6 + k]; M = M * E
    P = lambda s: tuple(int(c) - 1 for c in s)

    def chart(u):
        expr = sum(int(F4[r[P(u)], j]) * sp.prod([M[i, s[i]] for i in range(4)])
                   for j, s in enumerate(perms) if F4[r[P(u)], j])
        return dict(sp.Poly(sp.expand(expr), *t).terms())

    C = {u: chart(u) for u in ("3412", "1432", "3214", "4231", "2143", "4321")}
    for u in ("1432", "3214", "2143", "4321"):
        assert all(v > 0 for v in C[u].values()), u
    mono = lambda e: "*".join(f"t{i + 1}" + (f"^{k}" if k > 1 else "") for i, k in enumerate(e) if k) or "1"
    half = lambda x, y: tuple((p + q) for p, q in zip(x, y))
    out = {}
    # 3412
    assert len(C["3412"]) == 1 and list(C["3412"].values()) == [1]
    (m,) = C["3412"]
    pr = [(x, y) for x in C["1432"] for y in C["3214"] if half(x, y) == tuple(2 * z for z in m)]
    assert pr
    out["3412"] = {"m": mono(m), "a (Imm1432)": mono(pr[0][0]), "b (Imm3214)": mono(pr[0][1])}
    # 4231
    assert len(C["4231"]) == 3 and all(v == 1 for v in C["4231"].values())
    used = []
    for m in sorted(C["4231"], reverse=True):
        pr = [(x, y) for x in C["2143"] for y in C["4321"] if half(x, y) == tuple(2 * z for z in m)]
        assert len(pr) >= 1
        used.append((m, pr[0]))
    A_ = {x for _, (x, _) in used}
    B_ = {y for _, (_, y) in used}
    # the pairs used must be distinct terms of the product (a1 + a2)(b1 + b2)
    assert len({(x, y) for _, (x, y) in used}) == 3 and len(A_) <= 2 and len(B_) <= 2
    out["4231"] = [{"m": mono(m), "a (Imm2143)": mono(x), "b (Imm4321)": mono(y)} for m, (x, y) in used]
    out["verified"] = True
    return out


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _pattern(w, I):
    """0: other, 1: 3412, 2: 4231 (positions I sorted)."""
    a, b, c, d = w[I[0]], w[I[1]], w[I[2]], w[I[3]]
    if c < d and d < a and a < b:
        return 1
    if d < b and b < c and c < a:
        return 2
    return 0


@njit(parallel=True, cache=True)
def _occurrences(perms, combos, maxocc, occ, nocc):
    """For each w: up to maxocc occurrences of 3412 (listed first), then of 4231."""
    N = perms.shape[0]
    for wr in prange(N):
        w = perms[wr]
        k = 0
        for want in (1, 2):
            for ci in range(combos.shape[0]):
                if k >= maxocc:
                    break
                if _pattern(w, combos[ci]) == want:
                    occ[wr, k] = ci
                    k += 1
        nocc[wr] = k


@njit(parallel=True, cache=True)
def _g_vectors(task_w, task_c, combos, perms, perms4, permsc, factc, Fc, Ld, Rd, fact, start, size, xs,
               pid, val1, out):
    """out[t, s] = sum over sigma(I) = T with st(sigma|_I) = s of P_{sigma,w}(1) * f_{w_c}(st(sigma|_{I^c}))."""
    n = perms.shape[1]
    m = n - 4
    for t in prange(task_w.shape[0]):
        wr = task_w[t]
        w = perms[wr]
        I = combos[task_c[t]]
        inI = np.zeros(n, np.bool_)
        for a in range(4):
            inI[I[a]] = True
        Ic = np.empty(m, np.int64)
        k = 0
        for i in range(n):
            if not inI[i]:
                Ic[k] = i
                k += 1
        T = np.empty(4, np.int64)
        for a in range(4):
            T[a] = w[I[a]]
        T.sort()
        Tc = np.empty(m, np.int64)
        isT = np.zeros(n, np.bool_)
        for a in range(4):
            isT[T[a]] = True
        k = 0
        for v in range(n):
            if not isT[v]:
                Tc[k] = v
                k += 1
        # rank of w_c = st(w|_{I^c})
        wc = np.empty(m, np.int64)
        for b in range(m):
            c = 0
            for b2 in range(m):
                if w[Ic[b2]] < w[Ic[b]]:
                    c += 1
            wc[b] = c
        rc = _rank(wc, factc) if m > 0 else 0
        sigma = np.empty(n, np.int64)
        pb = np.empty(n, np.int64)
        vb = np.empty(n, np.int64)
        TT = np.zeros((n, n), np.int64)
        vb_top = np.zeros(n, np.int64)
        buf = np.empty(n, np.int64)
        lo = start[wr]
        hi = lo + size[wr]
        for s2 in range(permsc.shape[0]):
            coef = Fc[rc, s2]
            if coef == 0:
                continue
            for b in range(m):
                sigma[Ic[b]] = Tc[permsc[s2, b]]
            for s1 in range(24):
                for a in range(4):
                    sigma[I[a]] = T[perms4[s1, a]]
                key = _canon_rank(sigma, Ld[wr], Rd[wr], fact, pb, vb, TT, vb_top, buf)
                j = _bsearch(xs, lo, hi, key)
                if j >= 0:
                    out[t, s1] += val1[pid[j]] * coef


@njit(cache=True)
def _inv(a):
    c = 0
    for i in range(a.shape[0]):
        for j in range(i + 1, a.shape[0]):
            if a[i] > a[j]:
                c += 1
    return c


@njit(parallel=True, cache=True)
def _local_certificates(ws, combos, perms, perms4, fF, fG, Ld, Rd, fact, start, size, xs, pid, val1, out):
    """Proved certificates that use only P_{sigma,w}(1) for sigma agreeing with w off I (see the docstring).

    For each w in ws and every occurrence I of 3412 / 4231, with L[s] = P_{sigma(s, w_c), w}(1):
      bound:  3412: L[1432] + L[3214] <= 3;   4231: L[2143] == 2       (Billey-Braden upper bound)
      local:  sum_s f_W(s) L[s] < 0                                   (monomial lift, no Billey-Braden)
      tight:  3412: (i1 i3), (i2 i4) are Bruhat covers;  4231: 4231 -> 2143 lowers the length by exactly 3
    out[t] = flags: bit 0 has3412, bit 1 has4231, bits 2/3/4 bound/local/tight via 3412, bits 5/6/7 via 4231.
    """
    n = perms.shape[1]
    r1432, r3214, r2143 = 5, 14, 7          # lex ranks in S_4 of 1432, 3214, 2143 (asserted by the driver)
    for t in prange(ws.shape[0]):
        wr = ws[t]
        w = perms[wr]
        lw = _inv(w)
        sigma = np.empty(n, np.int64)
        pb = np.empty(n, np.int64)
        vb = np.empty(n, np.int64)
        TT = np.zeros((n, n), np.int64)
        vb_top = np.zeros(n, np.int64)
        buf = np.empty(n, np.int64)
        T = np.empty(4, np.int64)
        L = np.zeros(24, np.int64)
        lo = start[wr]
        hi = lo + size[wr]
        flags = 0
        for ci in range(combos.shape[0]):
            I = combos[ci]
            pat = _pattern(w, I)
            if pat == 0:
                continue
            flags |= 1 if pat == 1 else 2
            for a in range(4):
                T[a] = w[I[a]]
            T.sort()
            for s1 in range(24):
                for i in range(n):
                    sigma[i] = w[i]
                for a in range(4):
                    sigma[I[a]] = T[perms4[s1, a]]
                key = _canon_rank(sigma, Ld[wr], Rd[wr], fact, pb, vb, TT, vb_top, buf)
                j = _bsearch(xs, lo, hi, key)
                L[s1] = val1[pid[j]] if j >= 0 else 0
            fW = fF if pat == 1 else fG
            loc = 0
            for s1 in range(24):
                loc += fW[s1] * L[s1]
            sh = 2 if pat == 1 else 5
            if loc < 0:
                flags |= 1 << (sh + 1)
            if pat == 1:
                if L[r1432] + L[r3214] <= 3:
                    flags |= 1 << sh
                # covers: no point strictly inside the rectangle spanned by (i1, w i1), (i3, w i3), resp. (i2, i4)
                tight = True
                for (p, q) in ((I[0], I[2]), (I[1], I[3])):
                    for k in range(p + 1, q):
                        if w[q] < w[k] and w[k] < w[p]:
                            tight = False
                if tight:
                    flags |= 1 << (sh + 2)
            else:
                if L[r2143] == 2:
                    flags |= 1 << sh
                for i in range(n):
                    sigma[i] = w[i]
                A_, B_, C_, D_ = w[I[0]], w[I[1]], w[I[2]], w[I[3]]
                sigma[I[0]] = B_
                sigma[I[1]] = D_
                sigma[I[2]] = A_
                sigma[I[3]] = C_
                if lw - _inv(sigma) == 3:
                    flags |= 1 << (sh + 2)
        out[t] = flags


@njit(parallel=True, cache=True)
def _lemma_s(perms, Ld, Rd, fact, start, size, xs, pid, val1, out):
    """Lemma S on every inversion: out[wr] = (#inversions with R decreasing and P = 1, #R not decreasing and P > 1,
    #violations).  R = points of w strictly inside the rectangle (i, j) x (w(j), w(i)); x = w t_ij."""
    N, n = perms.shape
    for wr in prange(N):
        w = perms[wr]
        x = np.empty(n, np.int64)
        pb = np.empty(n, np.int64)
        vb = np.empty(n, np.int64)
        TT = np.zeros((n, n), np.int64)
        vb_top = np.zeros(n, np.int64)
        buf = np.empty(n, np.int64)
        lo = start[wr]
        hi = lo + size[wr]
        for i in range(n):
            for j in range(i + 1, n):
                if w[i] < w[j]:
                    continue
                dec = True
                last = n + 1
                for k in range(i + 1, j):
                    if w[j] < w[k] and w[k] < w[i]:
                        if w[k] > last:
                            dec = False
                        last = w[k]
                for k in range(n):
                    x[k] = w[k]
                x[i] = w[j]
                x[j] = w[i]
                key = _canon_rank(x, Ld[wr], Rd[wr], fact, pb, vb, TT, vb_top, buf)
                jj = _bsearch(xs, lo, hi, key)
                one = jj >= 0 and val1[pid[jj]] == 1
                if dec and one:
                    out[wr, 0] += 1
                elif (not dec) and (not one):
                    out[wr, 1] += 1
                else:
                    out[wr, 2] += 1


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--lemma", default="0", help="'all' or a number of sampled 321-avoiding occurrences")
    ap.add_argument("--maxocc", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bound", action="store_true", help="also run the proved local certificates")
    ap.add_argument("--lemma-s", action="store_true", help="check Lemma S on every inversion of every w")
    ap.add_argument("--out", default=None, help="output JSON (default results/pattern_lifting_n{n}.json)")
    a = ap.parse_args(argv)
    n = a.n
    t0 = time.time()
    log = lambda *s: print(f"[{time.time() - t0:7.0f}s]", *s, flush=True)
    res = {"n": n}

    res["witness_certificates"] = certify_witnesses()
    res["n4_inequalities"] = n4_inequalities()
    F4 = imm_table(4)
    Fc = imm_table(n - 4)
    combos_w, wvec = witness_vectors(F4)
    perms4 = np.array(list(permutations(range(4))), dtype=np.int64)
    permsc = np.array(list(permutations(range(n - 4))), dtype=np.int64).reshape(-1, max(n - 4, 0))
    factc = np.array([factorial(i) for i in range(max(n - 4, 1) + 1)], dtype=np.int64)
    # KL decoding of the witnesses: W(C'_u) for u in S_4 must equal the intended coefficients
    E4 = ExtremalKL(4)
    dec = {}
    for k, v in wvec.items():
        dec[k] = {}
        for u in permutations(range(4)):
            tot = sum(int(v[r]) * (sum(E4.P(s, u)) if E4.P(s, u) else 0) for r, s in enumerate(permutations(range(4))))
            if tot:
                dec[k]["".join(str(c + 1) for c in u)] = tot
    res["witness_KL_decoding"] = dec
    log("witnesses certified and decoded:", dec)

    E = ExtremalKL(n, verbose=n >= 9)
    Tb = E.tables
    log(f"KL table: {E.npairs} extremal pairs")
    combos = np.array(list(combinations(range(n), 4)), dtype=np.int64)
    occ = np.full((Tb.N, a.maxocc), -1, dtype=np.int64)
    nocc = np.zeros(Tb.N, dtype=np.int64)
    _occurrences(Tb.perms, combos, a.maxocc, occ, nocc)
    sing = np.flatnonzero(nocc > 0)
    res["singular"] = int(sing.size)
    log(f"singular: {sing.size}")

    def g_for(ws, cs):
        out = np.zeros((ws.size, 24), dtype=np.int64)
        _g_vectors(ws.astype(np.int64), cs.astype(np.int64), combos, Tb.perms, perms4, permsc, factc, Fc,
                   E.Ld, E.Rd, Tb.fact, E.start, E.size, E.xs, E.pid, E.values_at_one, out)
        return out

    # Theorem 9.6: certify every singular w, trying occurrences in order (3412 first)
    todo = sing.copy()
    certified = np.zeros(Tb.N, dtype=bool)
    tries = {}
    for r in range(a.maxocc):
        todo = todo[nocc[todo] > r]
        if todo.size == 0:
            break
        cs = occ[todo, r]
        g = g_for(todo, cs)
        pats = np.array([_pattern(Tb.perms[w], combos[c]) for w, c in zip(todo, cs)])
        val = np.where(pats == 1, g @ wvec["3412"], g @ wvec["4231"])
        ok = val < 0
        certified[todo[ok]] = True
        tries[r + 1] = int(ok.sum())
        todo = todo[~ok]
        log(f"occurrence {r + 1}: certified {int(ok.sum())}, remaining {todo.size}")
    res["certified_by_occurrence"] = tries
    res["not_certified"] = ["".join(str(c + 1) for c in Tb.perms[w]) for w in todo[:50]]
    res["all_singular_certified"] = bool(todo.size == 0)
    log(f"all singular w certified non-isolated: {todo.size == 0}")

    # Lemma S: P_{w t_ij, w} = 1 iff the points inside the (i, j) rectangle are decreasing
    if a.lemma_s:
        cnt = np.zeros((Tb.N, 3), dtype=np.int64)
        _lemma_s(Tb.perms, E.Ld, E.Rd, Tb.fact, E.start, E.size, E.xs, E.pid, E.values_at_one, cnt)
        tot = cnt.sum(axis=0)
        res["lemma_S"] = {"decreasing_and_smooth": int(tot[0]), "not_decreasing_and_singular": int(tot[1]),
                          "violations": int(tot[2])}
        log("Lemma S:", res["lemma_S"])

    # Proved local certificates over all occurrences (bound / monomial lift / tight)
    if a.bound:
        perms4_list = [tuple(int(c) for c in p) for p in perms4]
        assert perms4_list.index((0, 3, 2, 1)) == 5 and perms4_list.index((2, 1, 0, 3)) == 14 \
            and perms4_list.index((1, 0, 3, 2)) == 7
        flags = np.zeros(sing.size, dtype=np.int64)
        _local_certificates(sing.astype(np.int64), combos, Tb.perms, perms4, wvec["3412"].astype(np.int64),
                            wvec["4231"].astype(np.int64), E.Ld, E.Rd, Tb.fact, E.start, E.size, E.xs, E.pid,
                            E.values_at_one, flags)
        bit = lambda b: (flags >> b) & 1 == 1
        bound = bit(2) | bit(5)
        local = bit(3) | bit(6)
        tight = bit(4) | bit(7)
        name = lambda r: "".join(str(c + 1) for c in Tb.perms[r])
        res["local_certificates"] = {
            "singular": int(sing.size),
            "bound": int(bound.sum()), "bound_via_3412": int(bit(2).sum()), "bound_via_4231": int(bit(5).sum()),
            "local": int(local.sum()), "tight": int(tight.sum()),
            "bound_or_local": int((bound | local).sum()),
            "not_bound_or_local": [name(r) for r in sing[~(bound | local)][:200]],
            "not_bound_or_local_count": int((~(bound | local)).sum()),
            "not_bound_contains3412": int((~bound & bit(0)).sum()),
            # 4231-containing w are settled by the epsilon-witness (Proposition 9.3); what is left:
            "contains3412_avoids4231": int((bit(0) & ~bit(1)).sum()),
            "contains3412_avoids4231_not_bound": int((bit(0) & ~bit(1) & ~bit(2)).sum()),
            "contains3412_avoids4231_not_tight": int((bit(0) & ~bit(1) & ~bit(4)).sum()),
            "contains3412_avoids4231_not_tight_list": [name(r) for r in sing[bit(0) & ~bit(1) & ~bit(4)][:200]],
            "contains3412_avoids4231_not_bound_list": [name(r) for r in sing[bit(0) & ~bit(1) & ~bit(2)][:200]],
        }
        log("local certificates:", {k: v for k, v in res["local_certificates"].items() if not k.endswith("list") and k != "not_bound_or_local"})

    # Locality Lemma on 321-avoiding pattern occurrences
    if a.lemma != "0":
        perms4_list = [tuple(p) for p in perms4]
        avoid321 = np.array([not any(p[i] > p[j] > p[k] for i in range(4) for j in range(i + 1, 4)
                                     for k in range(j + 1, 4)) for p in perms4_list])
        rng = np.random.default_rng(a.seed)
        if a.lemma == "all":
            W_, C_ = np.meshgrid(np.arange(Tb.N), np.arange(len(combos)), indexing="ij")
            W_, C_ = W_.ravel(), C_.ravel()
        else:
            k = int(a.lemma) * 3
            W_ = rng.integers(0, Tb.N, size=k)
            C_ = rng.integers(0, len(combos), size=k)
        st = np.array([perms4_list.index(tuple(np.argsort(np.argsort(Tb.perms[w][combos[c]])))) for w, c in zip(W_, C_)])
        keep = avoid321[st]
        W_, C_, st = W_[keep], C_[keep], st[keep]
        if a.lemma != "all":
            W_, C_, st = W_[:int(a.lemma)], C_[:int(a.lemma)], st[:int(a.lemma)]
        bad = 0
        chunk = 200_000
        for s in range(0, W_.size, chunk):
            g = g_for(W_[s:s + chunk], C_[s:s + chunk])
            A = g @ F4.T                                         # A[t, u]
            expect = np.zeros_like(A)
            expect[np.arange(A.shape[0]), st[s:s + chunk]] = 1
            bad += int((A != expect).any(axis=1).sum())
        res["lemma"] = {"occurrences_checked": int(W_.size), "violations": bad}
        log(f"Locality Lemma: {W_.size} occurrences of 321-avoiding patterns checked, violations {bad}")

    res["seconds"] = round(time.time() - t0, 1)
    with open(a.out or f"results/pattern_lifting_n{n}.json", "w") as f:
        json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
