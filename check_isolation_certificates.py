#!/usr/bin/env python3
"""Exact certificates for the TNN-isolation classification in degrees 4 and 5 (Theorem 9.2 of paper/haiman_n10.tex).

The mathematics
---------------
Fix the standard chart (all lower factors, then all upper factors, both words the first reduced word
of w_0 in :func:`monimm.tnn_cone.reduced_words`).  In it every KL immanant is a polynomial
Imm_w = sum_alpha kappa_alpha(w) t^alpha with kappa_alpha(w) in N, and N denotes the Newton polytope,
the convex hull of the exponents alpha with kappa_alpha != 0.  The isolation criterion (Theorem 9.1
of paper/haiman_n10.tex) says

    w is TNN-isolated  <=>  some vertex alpha of N has supp kappa_alpha = {w}.

So the classification needs two kinds of certificate, both checked here in exact integer or
rational arithmetic (no floating point is trusted; HiGHS is only used to *find* certificates):

* w isolated: an exponent alpha with supp kappa_alpha = {w} and an integer functional c with
  c.alpha > c.beta for every other exponent beta.  Then alpha is a vertex of N.
* w not isolated: for every exponent alpha with supp kappa_alpha = {w}, a proof that alpha is not a
  vertex, namely other exponents beta_i and rationals y_i > 0 with sum y_i = 1 and
  sum y_i beta_i = alpha.  Most alpha are the midpoint of two other exponents, which is stored as a
  pair; the rest come from a simplex LP whose basic solution is then solved exactly.

The output records the chart, every exponent (as a string of digits, one per chart parameter), and
for every w its status and certificate, so a reader can re-verify Theorem 9.2 from the file alone
with integer arithmetic.  ``--verify FILE`` does exactly that, rebuilding the chart expansion.

Usage
-----
    .venv/bin/python check_isolation_certificates.py --n 4 --json results/tnn_isolation_certificates_n4.json
    .venv/bin/python check_isolation_certificates.py --n 5 --json results/tnn_isolation_certificates_n5.json
    .venv/bin/python check_isolation_certificates.py --verify results/tnn_isolation_certificates_n5.json

n = 5 has 45,356 exponents in 20 parameters and takes a few minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog

from monimm.tnn_cone import KLChartData, is_singular, perm_name, standard_chart


_HASH = np.random.default_rng(2026).integers(1, 2 ** 63, size=64, dtype=np.uint64) | np.uint64(1)


def _code(E: np.ndarray) -> np.ndarray:
    """A 64-bit hash of each integer row (wrapping arithmetic); matches are re-checked exactly."""
    return (E.astype(np.int64).view(np.uint64) * _HASH[:E.shape[1]]).sum(axis=1, dtype=np.uint64)


def _exp_str(a) -> str:
    return "".join(str(int(x)) for x in a)


def random_vertex(P: np.ndarray, rows, rng, rounds: int = 200) -> tuple | None:
    """(row, integer functional) with a unique maximum at that row, for one of ``rows``, or None.

    Random integer directions only; cheap, and finds the isolating vertex of almost every w."""
    Pi = P.astype(np.int64)
    d = P.shape[1]
    rows = set(int(r) for r in rows)
    for _ in range(rounds):
        C = rng.integers(-1000, 1001, size=(d, 64))
        vals = Pi @ C
        for k in range(C.shape[1]):
            col = vals[:, k]
            top = np.flatnonzero(col == col.max())
            if top.size == 1 and int(top[0]) in rows:
                return int(top[0]), [int(x) for x in C[:, k]]
    return None


def vertex_certificate(P: np.ndarray, rows) -> tuple | None:
    """(row, integer functional) from an LP for a strictly separating functional, or None."""
    Pi = P.astype(np.int64)
    d = P.shape[1]
    for r in sorted(int(r) for r in rows):
        D = P[r] - np.delete(P, r, axis=0)
        obj = np.zeros(d + 1)
        obj[-1] = -1
        res = linprog(obj, A_ub=np.hstack([-D, np.ones((D.shape[0], 1))]), b_ub=np.zeros(D.shape[0]),
                      bounds=[(-1, 1)] * d + [(None, 1)], method="highs")
        if res.status != 0 or -res.fun <= 1e-9:
            continue
        for scale in (10 ** 3, 10 ** 5, 10 ** 7):
            c = np.round(res.x[:d] * scale).astype(np.int64)
            col = Pi @ c
            if col[r] > np.delete(col, r).max():
                return r, [int(x) for x in c]
    return None


def midpoint_certificate(P: np.ndarray, r: int, codes) -> dict | None:
    """Exponent r as the midpoint of two other exponents, or None."""
    Pi = P.astype(np.int64)
    a = Pi[r]
    sc, order = codes
    other = _code(2 * a - Pi)
    pos = np.clip(np.searchsorted(sc, other), 0, len(sc) - 1)
    for s in np.flatnonzero(sc[pos] == other):
        t = int(order[pos[s]])
        if s != r and t != r and (Pi[s] + Pi[t] == 2 * a).all():
            return {"midpoint": [_exp_str(Pi[s]), _exp_str(Pi[t])]}
    return None


def nonvertex_certificate(P: np.ndarray, r: int, codes) -> dict | None:
    """Exact proof that exponent r is a convex combination of the other exponents."""
    Pi = P.astype(np.int64)
    a = Pi[r]
    # midpoint: a = (b + c)/2 with b, c other exponents
    sc, order = codes
    other = _code(2 * a - Pi)
    pos = np.clip(np.searchsorted(sc, other), 0, len(sc) - 1)
    for s in np.flatnonzero(sc[pos] == other):
        t = int(order[pos[s]])
        if s != r and t != r and (Pi[s] + Pi[t] == 2 * a).all():
            return {"midpoint": [_exp_str(Pi[s]), _exp_str(Pi[t])]}
    # general case: simplex LP, then the basic solution solved exactly
    idx = np.array([s for s in range(P.shape[0]) if s != r])
    A = np.vstack([P[idx].T, np.ones(len(idx))])
    b = np.concatenate([P[r], [1.0]])
    res = linprog(np.zeros(len(idx)), A_eq=A, b_eq=b, bounds=[(0, None)] * len(idx), method="highs-ds")
    if res.status != 0:
        return None
    S = idx[res.x > 1e-12]
    M = [[Fraction(int(Pi[s, k])) for s in S] for k in range(P.shape[1])] + [[Fraction(1)] * len(S)]
    y = _solve_exact(M, [Fraction(int(x)) for x in a] + [Fraction(1)])
    if y is None or any(v < 0 for v in y):
        return None
    return {"convex": [[_exp_str(Pi[s]), str(v)] for s, v in zip(S, y) if v != 0]}


def _solve_exact(M, b):
    """Unique solution of M y = b over Q (M has independent columns), or None."""
    rows, cols = len(M), len(M[0])
    A = [M[i][:] + [b[i]] for i in range(rows)]
    piv = []
    r = 0
    for c in range(cols):
        p = next((i for i in range(r, rows) if A[i][c] != 0), None)
        if p is None:
            return None
        A[r], A[p] = A[p], A[r]
        for i in range(rows):
            if i != r and A[i][c] != 0:
                f = A[i][c] / A[r][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[r])]
        piv.append(c)
        r += 1
    if any(A[i][cols] != 0 for i in range(r, rows)):
        return None
    return [A[i][cols] / A[i][i] for i in range(cols)]


def certify(n: int, seed: int = 0, log=print) -> dict:
    t0 = time.time()
    data = KLChartData(n)
    chart = standard_chart(n)
    K, P = data.chart_coefficients(chart)
    Ki = K.astype(np.int64)
    nz = (Ki != 0).sum(axis=1)
    allc = _code(P.astype(np.int64))
    order = np.argsort(allc)
    codes = (allc[order], order)
    rng = np.random.default_rng(seed)
    out = {"n": n, "chart": [f"{s}{i + 1}" for s, i in chart], "exponents": len(P), "perms": {}}
    for j, w in enumerate(data.perms):
        rows = np.flatnonzero((nz == 1) & (Ki[:, j] != 0))
        cert = random_vertex(P, rows, rng) if rows.size else None
        mids = {}
        if cert is None and rows.size:
            # exponents that are midpoints are not vertices; only the others can isolate w
            mids = {int(r): midpoint_certificate(P, int(r), codes) for r in rows}
            cert = vertex_certificate(P, [r for r, m in mids.items() if m is None])
        if cert is not None:
            r, c = cert
            out["perms"][perm_name(w)] = {"isolated": True, "exponent": _exp_str(P[r]), "functional": c}
            continue
        proofs = []
        for r in rows:
            pr = mids.get(int(r)) or nonvertex_certificate(P, int(r), codes)
            if pr is None:
                raise RuntimeError(f"{perm_name(w)}: no certificate either way at exponent {_exp_str(P[r])}")
            proofs.append({"exponent": _exp_str(P[r]), **pr})
        log(f"  {perm_name(w)}: not isolated, {len(proofs)} exponents, "
            f"{sum('convex' in p for p in proofs)} need a general convex combination")
        out["perms"][perm_name(w)] = {"isolated": False, "single_support_exponents": len(proofs),
                                      "non_vertex": proofs}
    iso = sorted(k for k, v in out["perms"].items() if v["isolated"])
    non = sorted(k for k, v in out["perms"].items() if not v["isolated"])
    out["isolated"] = len(iso)
    out["non_isolated"] = non
    out["non_isolated_equals_singular"] = non == sorted(perm_name(w) for w in data.perms if is_singular(w))
    out["seconds"] = round(time.time() - t0, 1)
    log(f"n = {n}: {len(iso)} isolated, {len(non)} not; non-isolated == singular: "
        f"{out['non_isolated_equals_singular']} ({out['seconds']} s)")
    return out


def verify(cert: dict, log=print) -> bool:
    """Re-check every certificate in exact arithmetic from a fresh chart expansion."""
    n = cert["n"]
    data = KLChartData(n)
    chart = standard_chart(n)
    assert [f"{s}{i + 1}" for s, i in chart] == cert["chart"], "different chart"
    K, P = data.chart_coefficients(chart)
    Pi = P.astype(np.int64)
    Ki = K.astype(np.int64)
    exps = {_exp_str(p): i for i, p in enumerate(Pi)}
    assert len(exps) == cert["exponents"]
    names = [perm_name(w) for w in data.perms]
    for j, name in enumerate(names):
        c = cert["perms"][name]
        if c["isolated"]:
            r = exps[c["exponent"]]
            assert set(np.flatnonzero(Ki[r])) == {j}, name
            vals = Pi @ np.array(c["functional"], dtype=np.int64)
            assert vals[r] > np.delete(vals, r).max(), name
        else:
            rows = {_exp_str(Pi[r]) for r in np.flatnonzero(((Ki != 0).sum(axis=1) == 1) & (Ki[:, j] != 0))}
            assert rows == {p["exponent"] for p in c["non_vertex"]}, name
            for p in c["non_vertex"]:
                a = [Fraction(int(x)) for x in p["exponent"]]
                if "midpoint" in p:
                    combo = [(p["midpoint"][0], Fraction(1, 2)), (p["midpoint"][1], Fraction(1, 2))]
                else:
                    combo = [(e, Fraction(y)) for e, y in p["convex"]]
                assert all(e in exps and e != p["exponent"] and y > 0 for e, y in combo), name
                assert sum(y for _, y in combo) == 1, name
                for k in range(len(a)):
                    assert sum(y * int(e[k]) for e, y in combo) == a[k], name
    log(f"n = {n}: all {len(names)} certificates verified exactly")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    ap.add_argument("--verify")
    a = ap.parse_args(argv)
    if a.verify:
        with open(a.verify) as f:
            return 0 if verify(json.load(f)) else 1
    out = certify(a.n, a.seed)
    verify(out)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
