#!/usr/bin/env python3
"""Check 12 of paper/haiman_n10.tex, all characters: the graded traces against Hoefsmit's seminormal form.

``check_induced_traces.py`` tests the Geck-Pfeiffer traces phi^lambda_q(T_v) of the graded engine at every
v in S_10, but only through the six induced modules M^mu of dimension <= 252, i.e. six linear combinations
of the 42 trace columns.  Here every column is tested.  For each mu |- n and q = 2, 3 the irreducible
character chi^mu_q(T_v) is computed for all n! permutations v from Hoefsmit's seminormal form
(:mod:`monimm.seminormal_traces`: explicit matrices, built from the tableau combinatorics alone, whose
quadratic and braid relations are checked first), and compared mod p with

    sum_lambda K_{mu lambda} phi^lambda_q(T_v)        (K the Kostka matrix, s_mu = sum K_{mu lambda} m_lambda),

the right side from the dumped traces.  Since K is unitriangular, agreement for all mu at every v is
agreement of all 42 columns phi^lambda_q(T_v) at every v.  No class polynomial, no character formula and
no double-coset sum is used.

Data written (the table is new data, so it is kept):
  * ``--out-npz`` (default results/seminormal_traces_n{n}_modp.npz): X[i, mu, v] = chi^mu_q(T_v) mod p at
    q = TRACE_QS[i], uint32, shapes in the order of ``monimm.combinatorics.partitions`` (stored as 'parts'),
    v in lexicographic order.  The irreducible Hecke character values at every T_v, not only at the
    class representatives.
  * ``--json``: per (mu, q) the number of mismatching v and the SHA-256 of the computed column.
  * ``--ckpt DIR``: one .npy per (mu, q) as it finishes, so an interrupted run resumes.

Usage
-----
    .venv/bin/python check_seminormal_traces.py --n 8 --fresh            # validate against a fresh engine run
    .venv/bin/python check_seminormal_traces.py --n 10 --ckpt results/seminormal_ckpt_n10 \\
        --json results/seminormal_traces_n10.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

from monimm.combinatorics import partitions
from monimm.ordering import kostka
from monimm.seminormal_traces import check_relations, chi_traces_all, standard_tableaux
from run_kl_graded import TRACE_PRIME, TRACE_QS

P = TRACE_PRIME


def log(msg, t0=[time.time()]):
    print(f"== [{time.time() - t0[0]:7.0f} s] {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--traces", default=None, help="traces npz (default results/kl_graded_n{n}_traces_modp.npz)")
    ap.add_argument("--fresh", action="store_true", help="recompute the traces with the graded engine (n <= 9)")
    ap.add_argument("--shapes", help="only these mu, e.g. '5,5 4,3,2,1' (default all)")
    ap.add_argument("--relations-nmax", type=int, default=7, help="check the relations densely for n <= this")
    ap.add_argument("--corrupt", action="store_true", help="negative control: perturb one trace entry first")
    ap.add_argument("--ckpt", help="directory for per-(mu, q) results (resumable)")
    ap.add_argument("--out-npz", default=None)
    ap.add_argument("--cut", type=int, default=6, help="tree depth at which tasks are split (more threads -> larger)")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    n = a.n
    parts = [tuple(x) for x in partitions(n)]

    # ---- the relations, densely, for every shape of size <= relations-nmax (and n itself if small)
    rel = {}
    for m in range(2, min(a.relations_nmax, n) + 1):
        for lam in partitions(m):
            for q in TRACE_QS:
                ok, msg = check_relations(tuple(lam), q, P)
                rel[f"{tuple(lam)} q={q}"] = msg
                assert ok, (lam, q, msg)
    log(f"seminormal relations verified for {len(rel)} (shape, q) with n <= {min(a.relations_nmax, n)}")

    # ---- the traces to test
    if a.fresh:
        from run_kl_graded import traces_mod_p
        from monimm.kl_extremal import ExtremalKL
        from monimm.kl_graded import hecke_traces
        E = traces_mod_p(hecke_traces(ExtremalKL(n))).astype(np.uint32)
        log(f"S_{n}: traces recomputed by the graded engine")
    else:
        z = np.load(a.traces or f"results/kl_graded_n{n}_traces_modp.npz")
        assert int(z["p"]) == P and tuple(z["q"]) == TRACE_QS
        assert [str(p) for p in parts] == list(z["parts"]), "shape order differs"
        E = z["E"]
        log(f"traces loaded: {E.shape}")
    if a.corrupt:
        E = E.copy()
        E[0, len(parts) // 2, 12345 % E.shape[2]] ^= 1
        log("negative control: one trace entry perturbed")

    K = np.array([[kostka(mu, lam) for lam in parts] for mu in parts], np.int64)
    assert all(K[i, i] == 1 for i in range(len(parts)))
    todo = [tuple(int(x) for x in s.split(",")) for s in a.shapes.split()] if a.shapes else list(parts)
    # largest first, so the long jobs start early and a resumed run has little left
    todo.sort(key=lambda mu: -len(standard_tableaux(mu)))
    if a.ckpt:
        os.makedirs(a.ckpt, exist_ok=True)

    out = {"n": n, "q": list(TRACE_QS), "prime": P, "traces": "fresh" if a.fresh else (a.traces or "default"),
           "relations_checked": len(rel), "results": {}}
    X = np.zeros((len(TRACE_QS), len(parts), math.factorial(n)), np.uint32)
    have = np.zeros((len(TRACE_QS), len(parts)), bool)
    for mu in todo:
        k = parts.index(mu)
        f = len(standard_tableaux(mu))
        for i, q in enumerate(TRACE_QS):
            name = f"{','.join(map(str, mu))}_q{q}"
            path = os.path.join(a.ckpt, name + ".npy") if a.ckpt else None
            t = time.time()
            if path and os.path.exists(path):
                got = np.load(path).astype(np.int64)
                src = "checkpoint"
            else:
                got = chi_traces_all(n, mu, q, P, cut=a.cut)
                src = "computed"
                if path:
                    np.save(path + ".tmp.npy", got.astype(np.uint32))
                    os.replace(path + ".tmp.npy", path)
            want = np.zeros(E.shape[2], np.int64)
            for l in range(len(parts)):
                if K[k, l]:
                    want = (want + int(K[k, l]) * E[i, l].astype(np.int64)) % P
            bad = int((got != want).sum())
            X[i, k] = got.astype(np.uint32)
            have[i, k] = True
            out["results"][f"{mu} q={q}"] = {"f": f, "mismatching_v": bad,
                                             "sha256": hashlib.sha256(X[i, k].tobytes()).hexdigest()}
            log(f"mu = {mu} (f = {f}), q = {q}: {bad} mismatching v of {len(got)} ({src}, {time.time() - t:.0f} s)")

    out["shapes_done"] = int(have.sum())
    out["pass"] = all(r["mismatching_v"] == 0 for r in out["results"].values())
    out["complete"] = bool(have.all())
    out["seconds"] = round(time.time() - log.__defaults__[0][0], 1)
    if out["complete"]:
        npz = a.out_npz or f"results/seminormal_traces_n{n}_modp.npz"
        np.savez(npz, X=X, q=np.array(TRACE_QS), p=np.int64(P), parts=np.array([str(p) for p in parts]))
        h = hashlib.sha256(open(npz, "rb").read()).hexdigest()
        out["npz"] = {"path": npz, "sha256": h}
        log(f"wrote {npz} (sha256 {h})")
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=1))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
