#!/usr/bin/env python3
"""Checks for Section 7.2 of the hooks paper: clean doubling steps for singular permutations.

Usage
-----
    .venv/bin/python notes/clean_steps.py --n 8                 # steps + coverage, ~1 min
    .venv/bin/python notes/clean_steps.py --n 9                 # reads results/big_n9.npz (or ext_n9.npz), ~2 min
    .venv/bin/python notes/clean_steps.py --n 7 --crowns        # Proposition 7.6 and Lemma 7.7, ~minutes

Writes results/clean_steps_n{n}.json.

Mathematics (details in Section 7.2 of the hooks paper)
-------------------------------------------------------
For s = s_l with sw < w < ws (w(l) < w(l+1), and the value l+1 stands left of the value l), Abreu-Nigro
eq. (3a) gives 2 c_w = c_ws + sum_{z in M(w,s)} mu(z,w) c_z, M(w,s) = {z < w : zs < z, mu(z,w) != 0}.
Each ch C'_z is Schur-positive and nonzero, so the step is *clean* (M empty) iff c_{.,ws} = 2 c_{.,w};
this column test is used below.  Left steps are right steps of the inverses (c_{.,w} = c_{.,w^{-1}}).

1. doubling: R = permutations reachable from smooth ones by clean steps (Theorem 7.4), and the
   singular w with no clean step down on either side (the reduction of Theorem 1.6(2)).
2. clean: for singular w, compare "clean" with "no bad cover at l", split by deg P_{e,w}
   (Proposition 7.8 predicts: deg 1 and no bad cover => clean).
3. explicit: R_ex = reachable by steps starting at a smooth w (Abreu-Nigro Prop. 3.1) or at a w with
   deg P_{e,w} = 1 and no bad cover.  Must be a subset of R.
4. --crowns (pure Bruhat order, no KL data): every z < w with l(w) - l(z) = 3 and four atoms (4-crown,
   equivalently mu(z,w) != 0) differs from w in four positions with w|D, z|D = 3412, 1324 or
   4231, 2143, its coatoms are the four local swaps of Proposition 7.6, and (Lemma 7.7) whenever
   w(l) < w(l+1), z(l) > z(l+1), w has a bad cover at l.

Smoothness is P_{e,w} = 1 (equivalently 3412- and 4231-avoidance).  P_{e,w} comes from the extremal KL
engine; c from kl_fast (n <= 8) or results/ext_n{n}.npz / big_n{n}.npz.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from itertools import permutations
from math import factorial

import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from monimm.kl_extremal import ExtremalKL  # noqa: E402


# ---------------------------------------------------------------------------
# Bruhat order and bad covers (0-indexed one-line notation)
# ---------------------------------------------------------------------------

@njit(cache=True)
def leq(x, y):
    """Bruhat order by the tableau criterion: x <= y iff #{k <= i : x(k) >= j} <= same for y."""
    n = x.shape[0]
    for i in range(n):
        for j in range(n):
            a = 0
            b = 0
            for k in range(i + 1):
                if x[k] >= j:
                    a += 1
                if y[k] >= j:
                    b += 1
            if a > b:
                return False
    return True


@njit(cache=True)
def is_cover(w, i, j):
    """For i < j with w(i) > w(j): w (i j) is covered by w iff no k in (i, j) has w(j) < w(k) < w(i)."""
    for k in range(i + 1, j):
        if w[j] < w[k] < w[i]:
            return False
    return True


@njit(cache=True)
def bad_cover(w, l):
    """Is there z = w (i j) covered by w with z(l) > z(l+1)?  Given w(l) < w(l+1), only
    (l+1, j) with w(j) < w(l), or (i, l) with w(i) > w(l+1), can produce one."""
    n = w.shape[0]
    i = l + 1
    for j in range(l + 2, n):
        if w[j] < w[i] and w[j] < w[l] and is_cover(w, i, j):
            return True
    j = l
    for i in range(0, l):
        if w[i] > w[j] and w[i] > w[l + 1] and is_cover(w, i, j):
            return True
    return False


def ranks(P):
    """Lexicographic ranks of the rows of P (the order of itertools.permutations)."""
    n = P.shape[1]
    r = np.zeros(P.shape[0], dtype=np.int64)
    for k in range(n):
        r += (P[:, k + 1:] < P[:, k:k + 1]).sum(axis=1) * factorial(n - 1 - k)
    return r


def pstr(w):
    return "".join(str(int(a) + 1) for a in w)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load(n):
    """perms (lex order), lengths, deg P_{e,w}, and the table C[lambda, w] = c_{lambda,w}."""
    P = np.array(list(permutations(range(n))), dtype=np.int64)
    E = ExtremalKL(n)
    e = tuple(range(n))
    deg = np.array([len(E.P(e, tuple(int(a) for a in p))) - 1 for p in P])
    if n <= 8:
        from monimm.kl_fast import FastKL
        f = FastKL(n)
        f.c_coefficients()
        C = f.C_scaled
    else:
        # the extremal and full-interval engines give bitwise the same table (results/big_n9.npz is shipped)
        path = next(f for f in (f"results/ext_n{n}.npz", f"results/big_n{n}.npz") if os.path.exists(f))
        C = np.load(path)["C"]
    length = np.zeros(len(P), dtype=np.int64)
    for i in range(n):
        length += (P[:, i + 1:] < P[:, i:i + 1]).sum(axis=1)
    return P, length, deg, C


def right_steps(P, l):
    """Rows w with s_l w < w < w s_l, and the ranks of w s_l."""
    pos = np.argsort(P, axis=1)
    Q = P.copy()
    Q[:, [l, l + 1]] = Q[:, [l + 1, l]]
    rows = np.flatnonzero((P[:, l] < P[:, l + 1]) & (pos[:, l + 1] < pos[:, l]))
    return rows, ranks(Q)[rows]


def step_checks(n, P, length, deg, C):
    N = len(P)
    sing = deg > 0
    Pinv = np.argsort(P, axis=1)
    inv_rank = ranks(Pinv)
    clean_up = [[] for _ in range(N)]      # clean steps w -> y
    ex_up = [[] for _ in range(N)]         # explicit steps w -> y (Abreu-Nigro Prop. 3.1 / Proposition 7.8)
    tab = Counter()
    prop3_fail = []
    for side, PP in ((0, P), (1, Pinv)):
        for l in range(n - 1):
            rows, ys = right_steps(PP, l)
            wr = rows if side == 0 else inv_rank[rows]
            yr = ys if side == 0 else inv_rank[ys]
            clean = (C[:, yr] == 2 * C[:, wr]).all(axis=0)
            for k in range(len(rows)):
                w, y = int(wr[k]), int(yr[k])
                if not sing[y]:
                    continue
                bc = bad_cover(PP[rows[k]], l)
                if clean[k]:
                    clean_up[w].append(y)
                if not sing[w] or (deg[w] == 1 and not bc):
                    ex_up[w].append(y)
                if side == 0 and sing[w]:
                    key = (f"deg {'1' if deg[w] == 1 else '>=2'}", "bad cover" if bc else "no bad cover",
                           "clean" if clean[k] else "not clean")
                    tab[" | ".join(key)] += 1
                    if deg[w] == 1 and not bc and not clean[k]:
                        prop3_fail.append((pstr(P[w]), l + 1))
                    if not sing[w] and not clean[k]:
                        raise AssertionError("Abreu-Nigro Prop. 3.1 violated: conventions wrong")

    def reach(up):
        ok = ~sing
        for w in np.argsort(length, kind="stable"):
            if ok[w]:
                for y in up[w]:
                    ok[y] = True
        return ok

    R, Rex = reach(clean_up), reach(ex_up)
    down = np.zeros(N, dtype=bool)          # singular y with a clean step down
    for w in range(N):
        for y in clean_up[w]:
            down[y] = True
    ns = int(sing.sum())
    return {
        "singular": ns,
        "R_clean_chains": int((R & sing).sum()),
        "R_explicit": int((Rex & sing).sum()),
        "R_explicit_subset_of_R": bool((Rex <= R).all()),
        "irreducible_singular": int((sing & ~down).sum()),
        "steps_from_singular_w": dict(tab),
        "prop3_counterexamples": prop3_fail[:20],
    }


# ---------------------------------------------------------------------------
# Crowns (pure Bruhat order)
# ---------------------------------------------------------------------------

SWAPS = {"3412": [(0, 2), (1, 2), (1, 3), (0, 3)], "4231": [(2, 3), (1, 3), (0, 2), (0, 1)]}
LOCAL_Z = {"3412": "1324", "4231": "2143"}


def patt(v):
    s = sorted(v)
    return "".join(str(s.index(x) + 1) for x in v)


def crown_checks(n, P, length):
    idx = {tuple(p): r for r, p in enumerate(P)}
    bylen = {}
    for r in range(len(P)):
        bylen.setdefault(int(length[r]), []).append(r)
    crowns = Counter()
    classification_ok = coatoms_ok = lemma4_ok = True
    lemma4_cases = 0
    for r in range(len(P)):
        w = P[r]
        for q in bylen.get(int(length[r]) - 3, []):
            z = P[q]
            if not leq(z, w):
                continue
            coat = sorted(y for y in bylen[int(length[r]) - 1] if leq(z, P[y]) and leq(P[y], w))
            if len(coat) != 4:
                continue
            D = [i for i in range(n) if z[i] != w[i]]
            pw = patt(w[D])
            crowns[(pw, patt(z[D]))] += 1
            if len(D) != 4 or pw not in SWAPS or patt(z[D]) != LOCAL_Z[pw]:
                classification_ok = False
                continue
            mine = []
            for a, b in SWAPS[pw]:
                y = w.copy()
                y[D[a]], y[D[b]] = y[D[b]], y[D[a]]
                mine.append(idx[tuple(y)])
            coatoms_ok &= sorted(mine) == coat
            for l in range(n - 1):
                if w[l] < w[l + 1] and z[l] > z[l + 1]:
                    lemma4_cases += 1
                    lemma4_ok &= bool(bad_cover(w, l))
    return {"four_crowns": {f"{a}->{b}": c for (a, b), c in crowns.items()},
            "classification_A": classification_ok, "coatoms_are_local_swaps": coatoms_ok,
            "lemma4_cases": lemma4_cases, "lemma4": lemma4_ok}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--crowns", action="store_true", help="also run the 4-crown checks (slow for n >= 7)")
    args = ap.parse_args(argv)
    n, t0 = args.n, time.time()
    P, length, deg, C = load(n)
    out = {"n": n}
    out.update(step_checks(n, P, length, deg, C))
    ns = out["singular"]
    print(f"n={n}: singular {ns}; clean chains {out['R_clean_chains']} ({100 * out['R_clean_chains'] / ns:.1f}%), "
          f"explicit {out['R_explicit']} ({100 * out['R_explicit'] / ns:.1f}%), subset {out['R_explicit_subset_of_R']}; "
          f"irreducible {out['irreducible_singular']} ({100 * out['irreducible_singular'] / ns:.1f}%); "
          f"Prop. 7.8 counterexamples {len(out['prop3_counterexamples'])}", flush=True)
    for k, v in sorted(out["steps_from_singular_w"].items()):
        print(f"    {k}: {v}")
    if args.crowns:
        out.update(crown_checks(n, P, length))
        print(f"  crowns {out['four_crowns']}; (A) {out['classification_A']}; coatoms {out['coatoms_are_local_swaps']}; "
              f"Lemma 7.7 on {out['lemma4_cases']} cases: {out['lemma4']}", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    with open(f"results/clean_steps_n{n}.json", "w") as f:
        json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
