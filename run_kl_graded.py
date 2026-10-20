"""Driver for :mod:`monimm.kl_graded`: Haiman's q-conjecture (J. AMS 1993, Conj. 2.1) for S_n.

Usage:  .venv/bin/python -u run_kl_graded.py --n 8 [--json results/kl_graded_n8.json] [--npz ...]
            [--traces-npz ...] [--codominant]

Computes every c_{lambda,w}(q) = phi^lambda_q(q^{l(w)/2} C'_w) and checks:
  * no int64 wraparound: every partial sum is below the bound of
    :func:`monimm.kl_graded.accumulation_bound`, which must be < 2^63;
  * c_{lambda,w}(1) equals the q = 1 coefficients of :mod:`monimm.kl_extremal`;
  * c_{lambda,w} = c_{lambda,w^{-1}} = c_{lambda,w_0 w w_0} as polynomials (not built in);
  * stability: c_{lambda u 1, w x e}(q) = c_{lambda,w}(q), against a fresh S_{n-1} run;
then reports the number of negative coefficients, of polynomials not symmetric about
q^{l(w)/2}, and of non-unimodal ones.  All arithmetic is exact (int64, with the degree
and overflow guards of the module); a non-zero count is a counterexample to that part of
the conjecture, to be re-checked independently.

Memory: the traces Tr and the output C each have n! * p(n) * (binom(n,2)+1) int64 entries
(about 56 GB each at n = 10), and the class polynomials as many again while the traces
are built.  The checks below work one shape at a time, so the peak is about
3 * 56 GB + the KL table at n = 10.

``--npz`` saves the full table C[lambda, w, k] (compressed; mostly zeros).  ``--traces-npz``
saves (uncompressed, uint32) E[i, lambda, v] = phi^lambda_q(T_v) mod TRACE_PRIME at q = TRACE_QS[i], for every v:
the Geck-Pfeiffer traces at the values used by the induced-module check (check 12 of
paper/haiman_n10.tex), so that check can be run later without the 56 GB polynomial array.
Lines starting with "== " are stage markers with
the elapsed time (the cloud watcher mails the latest one).
"""
import argparse, gc, json, time
import numpy as np
from numba import njit, prange
from monimm.kl_extremal import ExtremalKL
from monimm.kl_graded import hecke_traces, c_polynomials, check_properties, accumulation_bound

T0 = time.time()


TRACE_QS = (2, 3)
TRACE_PRIME = 2147483629          # prime < 2^31, as in check_inversion_exact.py


@njit(parallel=True, cache=True)
def _horner_mod_p(Tr, qs, p, E):
    for v in prange(Tr.shape[0]):
        for l in range(Tr.shape[1]):
            for i in range(qs.shape[0]):
                acc = 0
                for k in range(Tr.shape[2] - 1, -1, -1):
                    acc = (acc * qs[i] + Tr[v, l, k] % p) % p    # Python-style %, so acc in [0, p)
                E[i, l, v] = acc


def traces_mod_p(Tr, qs=TRACE_QS, p=TRACE_PRIME):
    """E[i, lambda, v] = sum_k Tr[v, lambda, k] qs[i]^k mod p, by Horner's rule (all values < 2^33)."""
    E = np.zeros((len(qs), Tr.shape[1], Tr.shape[0]), dtype=np.int64)
    _horner_mod_p(Tr, np.array(qs, dtype=np.int64), np.int64(p), E)
    return E


def stage(msg):
    print(f"== [{time.time() - T0:8.0f} s] {msg}", flush=True)


def run(n, traces_npz=None):
    t = time.time()
    stage(f"S_{n}: extremal KL table")
    K = ExtremalKL(n)
    stage(f"S_{n}: Hecke traces (class polynomials)")
    Tr = hecke_traces(K)
    K.timings["traces"] = time.time() - t - K.timings.get("tables", 0) - K.timings.get("extremal lists", 0) - K.timings.get("kl", 0)
    if traces_npz:
        stage(f"S_{n}: writing traces mod p to {traces_npz}")
        np.savez(traces_npz, E=traces_mod_p(Tr).astype(np.uint32), q=np.array(TRACE_QS), p=np.int64(TRACE_PRIME),
                            parts=np.array([str(tuple(x)) for x in K.tables.parts]))
    bound = accumulation_bound(K, Tr)
    stage(f"S_{n}: accumulation bound {bound:.3e} ({'ok' if bound < 2**63 else 'TOO LARGE'}); q-coefficients")
    C = c_polynomials(K, Tr)
    del Tr
    gc.collect()
    return K, C, bound


@njit(cache=True)
def _inverse_and_conjugate(perms, fact):
    """Lexicographic ranks of w^{-1} and w_0 w w_0 for every w (perms in lexicographic order)."""
    N, n = perms.shape
    inv = np.empty(N, np.int64)
    conj = np.empty(N, np.int64)
    q = np.empty(n, np.int64)
    c = np.empty(n, np.int64)
    for r in range(N):
        for i in range(n):
            q[perms[r, i]] = i
            c[i] = n - 1 - perms[r, n - 1 - i]
        for which in range(2):
            p = q if which == 0 else c
            rk = 0
            for i in range(n):
                smaller = 0
                for j in range(i + 1, n):
                    if p[j] < p[i]:
                        smaller += 1
                rk += smaller * fact[n - 1 - i]
            if which == 0:
                inv[r] = rk
            else:
                conj[r] = rk
    return inv, conj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--json")
    ap.add_argument("--npz", help="save the full table C[lambda, w, k]")
    ap.add_argument("--raw", action="store_true",
                    help="write --npz uncompressed (n = 10: zlib is ~30 min; compress afterwards with zstd -T0)")
    ap.add_argument("--traces-npz", help="save phi^lambda_q(T_v) mod p at q = 2, 3 for every v")
    ap.add_argument("--codominant", action="store_true",
                    help="also compare every codominant w with the chromatic quasisymmetric function (CHSS Cor. 7.5)")
    a = ap.parse_args()
    n = a.n
    K, C, bound = run(n, a.traces_npz)
    T = K.tables
    stage("check: q = 1 against kl_extremal")
    Cs, scale = K.c_coefficients_scaled()
    at_one = bool((C.sum(axis=2) * scale == Cs).all())
    del Cs
    stage("check: symmetries w -> w^-1, w0 w w0")
    inv, conj = _inverse_and_conjugate(T.perms.astype(np.int64), T.fact.astype(np.int64))
    assert inv[T.rank(tuple(range(n)))] == T.rank(tuple(range(n)))
    sym_inv = sym_conj = True
    for l in range(C.shape[0]):
        sym_inv &= bool((C[l] == C[l][inv]).all())
        sym_conj &= bool((C[l] == C[l][conj]).all())
    del inv, conj
    stage("check: stability against S_{n-1}")
    K1, C1, bound1 = run(n - 1)
    parts = [tuple(p) for p in T.parts]
    # w x e for w in S_{n-1}, in lexicographic order of S_n: append n-1 to the one-line notation
    emb = np.array([T.rank(tuple(int(x) for x in K1.tables.perms[r1]) + (n - 1,))
                    for r1 in range(K1.tables.N)], dtype=np.int64)
    stab = True
    for l1, lam in enumerate(K1.tables.parts):
        l = parts.index(tuple(sorted(tuple(lam) + (1,), reverse=True)))
        D1 = C1.shape[2]
        stab &= bool((C[l][emb, :D1] == C1[l1]).all()) and not C[l][emb, D1:].any()
    del K1, C1
    codom = None
    if a.codominant:
        from check_codominant_chromatic import compare
        stage("check: codominant w against chromatic quasisymmetric functions (independent of Hecke traces)")
        codom = compare(C, T, n, log=stage)
    stage("check: positivity, symmetry about q^{l/2}, unimodality")
    props = check_properties(C, T.length)
    hooks = [i for i, p in enumerate(parts) if len(p) == 1 or all(x == 1 for x in p[1:])]
    hook_props = check_properties(C, T.length, rows=hooks)
    out = {"n": n, "pairs_extremal": K.npairs, "distinct_P": int(K.polys.shape[0]),
           "max_coefficient": int(C.max()), "max_degree_used": int(np.flatnonzero(C.any(axis=(0, 1))).max()),
           "accumulation_bound": bound, "check_no_int64_overflow": bool(bound < 2 ** 63 and bound1 < 2 ** 63),
           "check_q1_matches_kl_extremal": at_one, "check_inverse_symmetry": sym_inv,
           "check_w0_conjugation_symmetry": sym_conj, "check_stability_vs_n_minus_1": stab,
           **props, "hooks": hook_props,
           "timings_s": {k: round(v, 1) for k, v in K.timings.items()}, "total_s": round(time.time() - T0, 1)}
    if codom is not None:
        out["codominant_chromatic_check"] = codom
    print(json.dumps(out, indent=1), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    if a.npz:
        stage(f"writing {a.npz}")
        (np.savez if a.raw else np.savez_compressed)(a.npz, C=C.astype(np.int64, copy=False), parts=np.array([str(p) for p in parts]))
    stage("DONE")


if __name__ == "__main__":
    main()
