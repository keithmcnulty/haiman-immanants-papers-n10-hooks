#!/usr/bin/env python3
"""Cross-check the extremal-pair KL engine against SageMath (the SageMath part of check 9 of
paper/haiman_n10.tex).

SageMath's ``KazhdanLusztigPolynomial`` (sage.combinat.kazhdan_lusztig) computes P_{x,w} in any
Coxeter group by the standard recursion, in code that shares nothing with monimm.  We compare it
with :class:`monimm.kl_extremal.ExtremalKL`:

* ``all`` mode (default for n <= 5): every pair (x, w) in S_n x S_n, including the pairs with
  x not <= w, where both sides must give P = 0.  This also tests the Bruhat order and the
  engine's lookup of a non-extremal x through its extremal representative.
* ``extremal`` mode (default for n >= 6): every extremal pair, i.e. every polynomial the engine
  stores (x maximal in W_{D_L(w)} x W_{D_R(w)}).

Conventions.  Our s_i (0-indexed) acts on the left by swapping the values i and i+1.  An element
is passed to Sage as the reduced word a_1 ... a_k (1-indexed letters) with
p = s_{a_1-1} ... s_{a_k-1}, obtained by repeatedly stripping a left descent, exactly as in
cross_check_coxeter3.py, and rebuilt there with ``W.from_reduced_word``.  Sage's ``P`` is a
polynomial in q, and the engine's coefficient lists are compared with its coefficients.

Two steps, so that SageMath and the repository's environment never have to share an interpreter:

    .venv/bin/python cross_check_sage.py --export 5 --out pairs_n5.json      # repository venv
    sage -python cross_check_sage.py --compare pairs_n5.json --json sage_n5.json

The second step imports only the standard library and Sage.  ``sage -python`` can be any Python
with SageMath importable (a Sage install, or the modular passagemath-combinat and
passagemath-groups wheels from PyPI).  ``python -m monimm.verify`` runs both steps when it finds
Sage ($SAGE_PYTHON, else ``sage -python`` if ``sage`` is on the PATH) for n = 2, ..., 6.
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def reduced_word(p):
    """Letters a_1..a_k (1-indexed) with p = s_{a_1-1} ... s_{a_k-1}; as in cross_check_coxeter3.py."""
    p = list(p)
    n = len(p)
    word = []
    while True:
        pos = {v: i for i, v in enumerate(p)}
        i = next((i for i in range(n - 1) if pos[i] > pos[i + 1]), None)
        if i is None:
            return word
        word.append(i + 1)
        a, b = pos[i], pos[i + 1]           # s_i p: swap the values i and i+1
        p[a], p[b] = i + 1, i


def export(n, mode):
    """The engine's P_{x,w} as [one-line x, one-line w, coefficients] rows (repository venv)."""
    from monimm.kl_extremal import ExtremalKL
    E = ExtremalKL(n)
    T = E.tables
    perms = [tuple(int(a) for a in p) for p in T.perms]
    rows = []
    if mode == "all":
        for w in perms:
            for x in perms:
                rows.append([x, w, E.P(x, w)])
    else:
        for r, w in enumerate(perms):
            for j in range(E.start[r], E.start[r] + E.size[r]):
                x = perms[int(E.xs[j])]
                rows.append([x, w, E.P(x, w)])
    return {"n": n, "mode": mode, "engine": "monimm/kl_extremal.py", "pairs": rows}


def compare(data):
    """Recompute every row with SageMath and count the disagreements (Sage interpreter)."""
    try:
        from sage.all import WeylGroup, LaurentPolynomialRing, ZZ, version
    except ImportError:                       # modular passagemath wheels
        from sage.all__sagemath_combinat import LaurentPolynomialRing, ZZ
        from sage.all__sagemath_groups import WeylGroup
        try:
            from sage.version import version
        except ImportError:
            version = "unknown"
    from sage.combinat.kazhdan_lusztig import KazhdanLusztigPolynomial
    t0 = time.time()
    n = data["n"]
    if n == 1:
        bad = [r for r in data["pairs"] if r[2] != [1]]
        return {"n": 1, "mode": data["mode"], "pairs_checked": len(data["pairs"]), "mismatches": len(bad),
                "examples": bad[:5], "sage_version": str(version() if callable(version) else version), "seconds": 0.0}
    W = WeylGroup(["A", n - 1], prefix="s")
    q = LaurentPolynomialRing(ZZ, "q").gen()
    KL = KazhdanLusztigPolynomial(W, q)
    elt = {}

    def g(p):
        p = tuple(p)
        if p not in elt:
            elt[p] = W.from_reduced_word(reduced_word(p))
        return elt[p]

    bad = []
    for x, w, ours in data["pairs"]:
        P = KL.P(g(x), g(w))
        d = {int(e): int(c) for e, c in P.dict().items() if c != 0}   # a Laurent polynomial in q
        if any(e < 0 for e in d):
            theirs = ["negative power of q", sorted(d.items())]       # cannot match the engine
        else:
            theirs = [d.get(e, 0) for e in range(max(d) + 1)] if d else []
        if theirs != list(ours):
            bad.append({"x": x, "w": w, "engine": ours, "sage": theirs})
    return {"n": n, "mode": data["mode"], "pairs_checked": len(data["pairs"]), "mismatches": len(bad),
            "examples": bad[:5], "sage_version": str(version() if callable(version) else version),
            "seconds": round(time.time() - t0, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", type=int, metavar="N", help="write the engine's P_{x,w} for S_N (repository venv)")
    ap.add_argument("--mode", choices=["all", "extremal"], help="default: all for N <= 5, extremal otherwise")
    ap.add_argument("--out", help="with --export: the file to write")
    ap.add_argument("--compare", metavar="FILE", help="recompute the rows of FILE with SageMath (Sage interpreter)")
    ap.add_argument("--json", help="with --compare: write a summary here")
    a = ap.parse_args(argv)
    if a.export is not None:
        data = export(a.export, a.mode or ("all" if a.export <= 5 else "extremal"))
        with open(a.out, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"n = {a.export}: {len(data['pairs'])} pairs ({data['mode']}) written to {a.out}")
        return 0
    if a.compare:
        with open(a.compare) as f:
            out = compare(json.load(f))
        print(json.dumps(out, indent=1))
        if a.json:
            with open(a.json, "w") as f:
                json.dump(out, f, indent=1)
        return 0 if out["mismatches"] == 0 else 1
    ap.error("give --export N --out FILE or --compare FILE")


if __name__ == "__main__":
    sys.exit(main())
