#!/usr/bin/env python3
"""Kazhdan-Lusztig positivity of monomial immanants for S_n with the extremal-pair engine.

Intended for n = 10 (1,252,173,030 extremal pairs, ~13 GB).  n <= 9 runs in minutes and
is the acceptance test: the c_{lambda,w} must equal results/big_n9.npz.

Usage
-----
    python run_kl_extremal.py --n 9 --out results/ext_n9 --compare results/big_n9.npz
    python run_kl_extremal.py --n 10 --out results/ext_n10

Writes OUT.npz (all c_{lambda,w}, int64, plus the parts) and OUT.json (a summary).
The checks (see the docstring of monimm/kl_extremal.py for the mathematics):

1. KL sanity: P(0) = 1, degree bound and P_{w,w} = 1 (inside the engine), non-negative
   coefficients.  The double coset sums are exact (hit counts = coset sizes, inside the engine).
2. J_mu values: c_{lambda, w_0(alpha)} = alpha! delta_{lambda, sort(alpha)} for every
   composition alpha of n, where w_0(alpha) is the longest element of the Young subgroup.
3. Symmetries c_{lambda,w} = c_{lambda,w^{-1}} = c_{lambda,w_0 w w_0}.  The recursion always
   uses a left descent, so it does not build these in.
4. Sampled KL inversion identity: the coefficient of x^u in sum_w c_{lambda,w} Imm_w is
   g_lambda(u), for random u (exact mod 2^64; check_inversion_exact.py makes it exact over Z).
5. Two-column shapes vanish at every w containing 321 (a theorem).
6. Codominant w: c_{.,w} recomputed from the Hessenberg set {v : v(i) <= max w(1..i)},
   with no KL polynomials.
7. Coproduct: for every decomposable w = w1 x w2, c_{lambda,w} =
   sum_{mu cup nu = lambda} c_{mu,w1} c_{nu,w2}, using independent runs for S_m, m < n.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import resource
import sys
import time

import numpy as np
from numba import njit, prange

from monimm.kl_extremal import ExtremalKL, _rank
from run_kl_big import _contains_321, _contains_312, _hessenberg_vectors


@njit(parallel=True, cache=True)
def _symmetry_ranks(perms, fact):
    """Ranks of w^{-1} and w_0 w w_0."""
    N, n = perms.shape
    inv = np.zeros(N, np.int64)
    conj = np.zeros(N, np.int64)
    for r in prange(N):
        p = perms[r]
        q = np.empty(n, np.int64)
        for i in range(n):
            q[p[i]] = i
        inv[r] = _rank(q, fact)
        for i in range(n):
            q[i] = n - 1 - p[n - 1 - i]
        conj[r] = _rank(q, fact)
    return inv, conj


def coproduct_check(C, T, small):
    """Vectorised check 7.  small[m] = (tables, C_m) for m < n."""
    n, parts = T.n, T.parts
    pindex = {lam: l for l, lam in enumerate(parts)}
    perms = T.perms.astype(np.int64)
    mx = np.maximum.accumulate(perms, axis=1)
    # first split point j+1: max of w(0..j) equals j
    hits = mx[:, :n - 1] == np.arange(n - 1)
    has = hits.any(axis=1)
    first = np.where(has, hits.argmax(axis=1) + 1, 0)
    checked = bad = 0
    for k in range(1, n):
        rows = np.nonzero(first == k)[0]
        if rows.size == 0:
            continue
        T1, C1 = small[k]
        T2, C2 = small[n - k]
        f1 = np.array([math.factorial(k - 1 - i) for i in range(k)], dtype=np.int64)
        f2 = np.array([math.factorial(n - k - 1 - i) for i in range(n - k)], dtype=np.int64)
        a = perms[rows, :k]
        b = perms[rows, k:] - k
        # Lehmer ranks, vectorised
        r1 = sum(((a[:, i:i + 1] > a[:, i + 1:]).sum(axis=1)) * f1[i] for i in range(k))
        r2 = sum(((b[:, i:i + 1] > b[:, i + 1:]).sum(axis=1)) * f2[i] for i in range(n - k))
        pred = np.zeros((len(parts), rows.size), dtype=np.int64)
        for l1, mu in enumerate(T1.parts):
            x = C1[l1, r1]
            for l2, nu in enumerate(T2.parts):
                lam = tuple(sorted(mu + nu, reverse=True))
                pred[pindex[lam]] += x * C2[l2, r2]
        checked += pred.size
        bad += int((pred != C[:, rows]).sum())
    return {"checked": int(checked), "mismatches": int(bad)}


def j_test_check(C, T):
    """Check 2 over all compositions alpha of n."""
    n, parts = T.n, T.parts
    pindex = {lam: l for l, lam in enumerate(parts)}
    bad = 0
    for cuts in itertools.product([0, 1], repeat=n - 1):
        blocks, s = [], 0
        for i, c in enumerate(cuts):
            if c:
                blocks.append((s, i))
                s = i + 1
        blocks.append((s, n - 1))
        w = tuple(lo + hi - i for lo, hi in blocks for i in range(lo, hi + 1))
        alpha = [hi - lo + 1 for lo, hi in blocks]
        expect = np.zeros(len(parts), dtype=np.int64)
        expect[pindex[tuple(sorted(alpha, reverse=True))]] = math.prod(math.factorial(a) for a in alpha)
        bad += not np.array_equal(C[:, T.rank(w)], expect)
    return {"compositions": 2 ** (n - 1), "mismatches": bad}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=2000, help="u's for the sampled identity")
    ap.add_argument("--compare", help="npz with reference C (e.g. results/big_n9.npz)")
    ap.add_argument("--skip-small", action="store_true", help="skip the coproduct check")
    args = ap.parse_args(argv)
    n = args.n
    t0 = time.time()
    summary = {"n": n, "engine": "monimm/kl_extremal.py"}
    log = lambda *a: print(f"[{time.time() - t0:7.0f}s]", *a, flush=True)

    E = ExtremalKL(n, verbose=True)
    T = E.tables
    summary.update(extremal_pairs=E.npairs, distinct_polys=len(E.polys), sanity=E.sanity(),
                   max_poly_degree=int((E.polys != 0).sum(axis=1).max() - 1),
                   max_mu_list=E.max_mu)
    log(f"KL done: {E.npairs} extremal pairs, {len(E.polys)} distinct polynomials, "
        f"sanity {E.sanity()}, timings {E.timings}")
    C, scale = E.c_coefficients_scaled()
    assert scale == 1 or n < 10
    summary["scale"] = scale
    log(f"coefficients done, timings {E.timings}")
    parts = T.parts
    shapes = {}
    for l, lam in enumerate(parts):
        row = C[l]
        nz = row[row != 0]
        shapes[",".join(map(str, lam))] = {
            "support": int(nz.size), "min": int(nz.min()), "max": int(nz.max()),
            "negative": int((nz < 0).sum())}
    summary["shapes"] = shapes
    summary["all_nonnegative"] = bool((C >= 0).all())
    log(f"all c_(lambda,w) >= 0: {summary['all_nonnegative']}")

    if args.compare:
        ref = np.load(args.compare)
        assert [str(p) for p in ref["parts"]] == ["".join(map(str, p)) for p in parts]
        summary["equals_reference"] = bool(np.array_equal(ref["C"], C))
        log(f"equal to {args.compare}: {summary['equals_reference']}")

    summary["j_test_values"] = j_test_check(C, T)
    inv, conj = _symmetry_ranks(T.perms, T.fact)
    summary["symmetry_inverse"] = bool(np.array_equal(C, C[:, inv]))
    summary["symmetry_w0_conjugation"] = bool(np.array_equal(C, C[:, conj]))
    del inv, conj
    log(f"J_mu values {summary['j_test_values']}, symmetries "
        f"{summary['symmetry_inverse']}, {summary['symmetry_w0_conjugation']}")

    t = time.time()
    summary["sampled_identity"] = {"samples": min(args.samples, T.N),
                                   "pass": E.sampled_identity(C, samples=args.samples)}
    summary["timings_identity"] = round(time.time() - t, 1)
    log(f"sampled identity: {summary['sampled_identity']}")

    c321 = _contains_321(T.perms)
    two_col = [l for l, lam in enumerate(parts) if lam[0] <= 2]
    summary["two_column_vanish_on_321"] = bool(all(not C[l][c321].any() for l in two_col))
    del c321
    cod = np.nonzero(~_contains_312(T.perms))[0].astype(np.int64)
    H = _hessenberg_vectors(T.perms, cod, T.ctype, E.G)
    summary["codominant_hessenberg_check"] = bool(np.array_equal(H.T, C[:, cod]))
    summary["codominant_count"] = int(cod.size)
    log(f"two-column {summary['two_column_vanish_on_321']}, "
        f"codominant {summary['codominant_hessenberg_check']} ({cod.size})")

    if not args.skip_small:
        small = {}
        for m in range(1, n):
            Em = ExtremalKL(m)
            Cm, sm = Em.c_coefficients_scaled()
            assert sm == 1
            small[m] = (Em.tables, Cm)
            del Em
        summary["coproduct"] = coproduct_check(C, T, small)
        log(f"coproduct {summary['coproduct']}")

    summary["seconds"] = round(time.time() - t0, 1)
    summary["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    summary["timings"] = {k: round(v, 1) for k, v in E.timings.items()}
    np.savez_compressed(args.out + ".npz", C=C, parts=np.array(["".join(map(str, p)) for p in parts]))
    json.dump(summary, open(args.out + ".json", "w"), indent=1)
    ok = (summary["all_nonnegative"] and all(summary["sanity"].values())
          and summary["j_test_values"]["mismatches"] == 0
          and summary["symmetry_inverse"] and summary["symmetry_w0_conjugation"]
          and summary["sampled_identity"]["pass"]
          and summary["two_column_vanish_on_321"] and summary["codominant_hessenberg_check"]
          and summary.get("coproduct", {"mismatches": 0})["mismatches"] == 0
          and summary.get("equals_reference", True))
    log(json.dumps({k: v for k, v in summary.items() if k != "shapes"}, indent=1))
    log("ALL CHECKS PASS" if ok else "SOME CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
