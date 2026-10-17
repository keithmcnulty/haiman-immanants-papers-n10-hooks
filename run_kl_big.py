#!/usr/bin/env python3
"""Kazhdan-Lusztig positivity of monomial immanants for S_n with the parallel engine.

Intended for n = 9 on a large machine (>= 128 GB RAM).  n = 8 runs in seconds and is
the acceptance test: the output must agree with results/big_n8.npz.

Usage
-----
    python run_kl_big.py --n 8 --out results/big_n8            # ~10 s, 2 GB
    python run_kl_big.py --n 9 --out results/big_n9            # ~10-20 min, ~90 GB

Writes OUT.npz (all c_{lambda,w} as an int64 array, plus the parts) and OUT.json (a
summary).  The checks:

1. Exact identity: sum_w c_{lambda,w} Imm_w = Imm_lambda on all n! permutation
   monomials.  This also tests the KL inversion formula.
2. KL sanity: P(0) = 1, degree bound (inside the engine), non-negative coefficients.
3. Coproduct: for every decomposable w = w1 x w2, c_{lambda,w} =
   sum_{mu cup nu = lambda} c_{mu,w1} c_{nu,w2}, using independently computed S_m, m < n.
4. Two-column shapes vanish at every w containing 321 (a theorem; checks conventions).
5. Codominant w: c_{.,w} recomputed from the Hessenberg set {v : v(i) <= max w(1..i)},
   with no KL polynomials.
6. Per-vertex hook positivity (a conjecture): f^(a)_i(C'_w(1)) >= 0 for all a, i, w.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time

import numpy as np
from numba import njit, prange

from monimm.combinatorics import monomial_class_functions
from monimm.kl_big import BigKL


@njit(parallel=True, cache=True)
def _contains_321(perms):
    N, n = perms.shape
    out = np.zeros(N, dtype=np.bool_)
    for r in prange(N):
        p = perms[r]
        # 321 iff some j has a larger value before it and a smaller value after it
        found = False
        for j in range(1, n - 1):
            big = False
            for i in range(j):
                if p[i] > p[j]:
                    big = True
                    break
            if big:
                for k in range(j + 1, n):
                    if p[k] < p[j]:
                        found = True
                        break
            if found:
                break
        out[r] = found
    return out


@njit(parallel=True, cache=True)
def _contains_312(perms):
    N, n = perms.shape
    out = np.zeros(N, dtype=np.bool_)
    for r in prange(N):
        p = perms[r]
        found = False
        for i in range(n):
            for j in range(i + 1, n):
                if p[j] < p[i]:
                    for k in range(j + 1, n):
                        if p[j] < p[k] < p[i]:
                            found = True
                            break
                if found:
                    break
            if found:
                break
        out[r] = found
    return out


@njit(parallel=True, cache=True)
def _hessenberg_vectors(perms, cod, ctype, G):
    """For codominant w: sum over {v : v(i) <= m(i)} of G[:, ctype(v)]."""
    N, n = perms.shape
    L = G.shape[0]
    out = np.zeros((cod.shape[0], L), dtype=np.int64)
    for k in prange(cod.shape[0]):
        w = perms[cod[k]]
        m = np.empty(n, dtype=np.int64)
        mx = -1
        for i in range(n):
            if w[i] > mx:
                mx = w[i]
            m[i] = mx
        for r in range(N):
            v = perms[r]
            ok = True
            for i in range(n):
                if v[i] > m[i]:
                    ok = False
                    break
            if ok:
                for l in range(L):
                    out[k, l] += G[l, ctype[r]]
    return out


@njit(parallel=True, cache=True)
def _cycle_lengths(perms):
    N, n = perms.shape
    cl = np.zeros((N, n), dtype=np.int8)
    for r in prange(N):
        p = perms[r]
        for i in range(n):
            L = 1
            j = p[i]
            while j != i:
                j = p[j]
                L += 1
            cl[r, i] = L
    return cl


@njit(parallel=True, cache=True)
def _vertex_sums(start, size, below, pid, val1, length, cl):
    """H[w, i, L] = sum_{x <= w, l_i(x) = L} sgn(x) P_{x,w}(1)."""
    N, n = cl.shape
    H = np.zeros((N, n, n + 1), dtype=np.int64)
    for w in prange(N):
        a, m = start[w], size[w]
        for t in range(m):
            x = below[a + t]
            v = val1[pid[a + t]]
            if length[x] % 2 == 1:
                v = -v
            for i in range(n):
                H[w, i, cl[x, i]] += v
    return H


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-small", action="store_true", help="skip the coproduct check")
    args = ap.parse_args(argv)
    n = args.n
    t0 = time.time()
    summary = {"n": n}
    log = lambda *a: print(*a, flush=True)

    big = BigKL(n, verbose=True)
    T = big.tables
    summary.update(npairs=big.npairs, distinct_polys=len(big.polys), sanity=big.sanity(),
                   max_poly_degree=int((big.polys != 0).sum(axis=1).max() - 1))
    log(f"KL done: {big.npairs} pairs, {len(big.polys)} distinct polynomials, sanity {big.sanity()}, "
        f"timings {big.timings}")
    C, scale = big.c_coefficients_scaled()          # raises unless the identity holds exactly
    assert scale == 1
    summary["identity_check"] = True
    parts = T.parts
    shapes = {}
    for l, lam in enumerate(parts):
        row = C[l]
        nz = row[row != 0]
        shapes["".join(map(str, lam)) if n < 10 else ",".join(map(str, lam))] = {
            "support": int(nz.size), "min": int(nz.min()), "max": int(nz.max()),
            "negative": int((nz < 0).sum())}
    summary["shapes"] = shapes
    summary["all_nonnegative"] = bool((C >= 0).all())
    log(f"coefficients: all non-negative = {summary['all_nonnegative']}")

    # 4. two-column support
    c321 = _contains_321(T.perms)
    two_col = [l for l, lam in enumerate(parts) if lam[0] <= 2]
    summary["two_column_vanish_on_321"] = bool(all(not C[l][c321].any() for l in two_col))

    # 5. codominant via Hessenberg sets
    _, g = monomial_class_functions(n)
    G = np.array([[int(x) for x in row] for row in g], dtype=np.int64)
    cod = np.nonzero(~_contains_312(T.perms))[0].astype(np.int64)
    H = _hessenberg_vectors(T.perms, cod, T.ctype, G)
    summary["codominant_hessenberg_check"] = bool(np.array_equal(H.T, C[:, cod]))
    summary["codominant_count"] = int(cod.size)

    # 6. per-vertex hook positivity
    V = _vertex_sums(big.start, big.size, big.below, big.pid, big.values_at_one, T.length,
                     _cycle_lengths(T.perms))
    tails = np.cumsum(V[:, :, ::-1], axis=2)[:, :, ::-1]
    neg = 0
    for a in range(1, n + 1):
        F = (-1) ** (a - 1) * tails[:, :, a]
        neg += int((F < 0).sum())
    summary["per_vertex_hook_negatives"] = neg
    del V, tails

    # 3. coproduct against independent smaller computations
    if not args.skip_small:
        small = {}
        for m in range(1, n):
            b = BigKL(m)
            Cm, _ = b.c_coefficients_scaled()
            small[m] = (b.tables, Cm)
        bad = checked = 0
        perms = T.perms
        mx = np.maximum.accumulate(perms, axis=1)
        for r in range(T.N):
            split = next((j + 1 for j in range(n - 1) if mx[r, j] == j), None)
            if split is None:
                continue
            w1 = tuple(int(x) for x in perms[r, :split])
            w2 = tuple(int(x) - split for x in perms[r, split:])
            T1, C1 = small[split]
            T2, C2 = small[n - split]
            r1, r2 = T1.rank(w1), T2.rank(w2)
            for l, lam in enumerate(parts):
                tot = 0
                for l1, mu in enumerate(T1.parts):
                    if C1[l1, r1] == 0:
                        continue
                    for l2, nu in enumerate(T2.parts):
                        if C2[l2, r2] and tuple(sorted(mu + nu, reverse=True)) == lam:
                            tot += int(C1[l1, r1]) * int(C2[l2, r2])
                checked += 1
                bad += tot != int(C[l, r])
        summary["coproduct"] = {"checked": checked, "mismatches": bad}

    summary["seconds"] = round(time.time() - t0, 1)
    summary["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    summary["timings"] = {k: round(v, 1) for k, v in big.timings.items()}
    np.savez_compressed(args.out + ".npz", C=C, parts=np.array(["".join(map(str, p)) for p in parts]))
    json.dump(summary, open(args.out + ".json", "w"), indent=1)
    ok = (summary["all_nonnegative"] and summary["two_column_vanish_on_321"]
          and summary["codominant_hessenberg_check"] and all(summary["sanity"].values())
          and summary.get("coproduct", {"mismatches": 0})["mismatches"] == 0)
    log(json.dumps({k: v for k, v in summary.items() if k != "shapes"}, indent=1))
    log("ALL CHECKS PASS" if ok else "SOME CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
