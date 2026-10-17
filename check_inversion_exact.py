#!/usr/bin/env python3
"""Exact (over Z) version of the sampled Kazhdan-Lusztig inversion check, check 4 of
run_kl_extremal.py, on a stored coefficient table.

The mathematics
---------------
Rhoades-Skandera define Imm_w(x) = sum_{v >= w} (-1)^{l(v)-l(w)} Q_{w,v}(1) x_{1,v(1)}...x_{n,v(n)},
with Q the inverse KL polynomials, Q_{w,v} = P_{w0 v, w0 w}.  If
Imm_lambda = sum_w c_{lambda,w} Imm_w, then comparing coefficients of the monomial x^u gives

    sum_{w <= u} (-1)^{l(u)-l(w)} P_{w0 u, w0 w}(1) c_{lambda,w} = g_lambda(u).

This identity is checked for random u, against the stored c_{lambda,w} and the KL
polynomials recomputed by the extremal-pair engine.  run_kl_extremal.py accumulates the
left side in int64, which wraps around, so its check holds only modulo 2^64.  Here each sum is
also accumulated modulo a prime p < 2^31, and the float64 bound
B = sum |P(1)| |c| on every partial sum is recorded.  If B < 2^62, the int64 sums cannot wrap,
and agreement is an identity over Z (see monimm.kl_extremal._sampled_identity_exact).

Usage
-----
    python check_inversion_exact.py --n 10 --coeffs results/ext_n10.npz --out results/inversion_exact_n10.json

n = 10 rebuilds the extremal KL table (~10 GB, about 25 minutes on 16 cores) and reuses the
stored C, so the coefficient stage is skipped.  The same seed as run_kl_extremal.py gives the
same 2000 u.
"""

from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np

from monimm.kl_extremal import ExtremalKL


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--coeffs", required=True, help="npz with C (from run_kl_extremal.py)")
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:7.0f}s]", *a, flush=True)

    z = np.load(args.coeffs)
    C = z["C"]
    E = ExtremalKL(args.n, verbose=True)
    log(f"KL done: {E.npairs} extremal pairs, {len(E.polys)} distinct polynomials")
    parts = ["".join(map(str, lam)) for lam in E.tables.parts]
    assert parts == [str(s) for s in z["parts"]], "stored table has a different shape order"
    G, scale = E.g_table()
    assert scale == 1, "stored C is unscaled only when scale == 1"
    res = E.sampled_identity_exact(C, samples=args.samples, seed=args.seed)
    res.update(n=args.n, coeffs=args.coeffs, seed=args.seed, seconds=round(time.time() - t0, 1),
               peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2))
    log(json.dumps(res, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if res["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
