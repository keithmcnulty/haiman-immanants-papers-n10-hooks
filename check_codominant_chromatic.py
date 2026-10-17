"""Independent check of Haiman's q-polynomials at every codominant w: c_{lam,w}(q) = [e_lam] X_{G_w}(x, q).

See monimm/chromatic.py for the statement (Clearman-Hyatt-Shelton-Skandera Cor. 7.5) and the method.  The Hecke side
is monimm.kl_graded.c_polynomials; the chromatic side uses no Hecke algebra.
Usage:  .venv/bin/python -u check_codominant_chromatic.py --n 9 [--json results/codominant_chromatic_n9.json]
"""
import argparse, json, time
import numpy as np
from monimm.kl_extremal import ExtremalKL
from monimm.kl_graded import c_polynomials
from monimm.chromatic import codominant_perms, e_coefficients


def compare(C, T, n, log=print):
    t = time.time()
    perms = codominant_perms(n)
    parts = [tuple(p) for p in T.parts]
    E = e_coefficients(n, perms, parts, D=C.shape[2])
    ranks = np.array([T.rank(p) for p in perms], np.int64)
    H = C[:, ranks, :].astype(np.int64)
    mism = np.flatnonzero((H != E).any(axis=(0, 2)))
    log(f"== codominant chromatic check, n = {n}: {len(perms)} codominant w x {len(parts)} lam; "
        f"mismatching w: {mism.size} ({time.time() - t:.0f} s)")
    return {"n": n, "codominant_w": len(perms), "shapes": len(parts), "mismatching_w": int(mism.size),
            "mismatch_examples": ["".join(str(a + 1) for a in perms[i]) for i in mism[:5]],
            "seconds": round(time.time() - t, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    K = ExtremalKL(a.n)
    C = c_polynomials(K)
    out = compare(C, K.tables, a.n)
    print(json.dumps(out, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
