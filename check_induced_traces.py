#!/usr/bin/env python3
"""Check 12 of paper/haiman_n10.tex at n = 10: the graded traces and table against induced Hecke modules.

``check_graded_induced.py`` does check 12 for n <= 6 with every module M^mu.  At n = 10 neither the
level-by-level matrices nor the big modules (dim M^{1^10} = 10!) fit, so this script checks what does fit,
with the depth-first kernel of :mod:`monimm.induced_traces`, against the files written by the S_10 rerun
(``cloud/run_kl_graded_n10_tables.sh``):

  * ``results/kl_graded_n10_traces_modp.npz``: E[i, lambda, v] = phi^lambda_q(T_v) mod p at q = 2, 3
    (p = 2147483629), from the Geck-Pfeiffer class polynomials of :mod:`monimm.kl_graded`;
  * ``results/kl_graded_n10.npz.zst``: the full table C[lambda, w, k] (zstd of an uncompressed npz;
    56 GB once decompressed, so it is streamed, never stored), with ``kl_graded_n10.npz.sha256``.

Part A (every v).  For each mu in a list of small modules and q = 2, 3:

    eta^mu_q(T_v) = sum_lambda N_{mu lambda} phi^lambda_q(T_v)   (mod p)   for all 10! permutations v,

the left side computed from the Hecke relations alone on M^mu, the right side from the dumped traces.
N_{mu lambda} counts non-negative integer matrices with row sums mu and column sums lambda.  This tests a
linear combination of the 42 trace columns per mu at every v; it cannot test all of them, since that would
need every M^mu.

Part B (sampled w, end to end).  For random non-codominant w (312-containing; the codominant w are already
checked independently against chromatic quasisymmetric functions, check 11), with P_{v,w} from du Cloux's
Coxeter3 (an independent C++ implementation):

    (B1)  C[lambda, w](q) = sum_{v <= w} P_{v,w}(q) phi^lambda_q(T_v)                for all 42 lambda,
    (B2)  sum_lambda N_{mu lambda} C[lambda, w](q) = sum_{v <= w} P_{v,w}(q) eta^mu_q(T_v)   for each mu,

mod p at q = 2, 3.  (B2) uses no class polynomial and no part of the graded engine except the stored
table: it is the n = 10 analogue of check 12.  (B1) checks the coefficient assembly of the engine for all
lambda, given the traces.

P_{v,w} for every v.  Coxeter3's ``extremals`` lists P_{x,w} for the x <= w that are maximal in their
double coset W_I x W_J, I = D_L(w), J = D_R(w).  Since P_{v,w} = P_{sv,w} = P_{vs',w} for s in I, s' in J
(Kazhdan-Lusztig 1979, (2.3.g)), P_{v,w} = P_{v*,w} with v* the maximum of W_I v W_J, and v <= w iff
v* <= w (w is the maximum of its own double coset).  v* is found by sorting values inside the I-blocks
and positions inside the J-blocks into decreasing order, repeated until stable (each pass that changes
anything raises the length, and the stable element has I, J among its descents, which characterises the
maximum).  If v* is not in the list, v is not below w.

Usage
-----
    .venv/bin/python check_induced_traces.py --n 10 --json results/induced_traces_n10.json
    .venv/bin/python check_induced_traces.py --n 8  --fresh      # recompute traces and table (n <= 9)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import time

import numpy as np
from numba import njit, prange

from check_graded_induced import _count
from cross_check_coxeter3 import parse_blocks, reduced_word, run_coxeter3, word_to_perm
from monimm.combinatorics import partitions
from monimm.induced_traces import eta_traces_all
from run_kl_graded import TRACE_PRIME, TRACE_QS

P = TRACE_PRIME
DEFAULT_MU = {10: "9,1 8,2 8,1,1 7,3 6,4 5,5"}


def log(msg, t0=[time.time()]):
    print(f"== [{time.time() - t0[0]:7.0f} s] {msg}", flush=True)


def lex_perms(n):
    """All permutations of 0..n-1 in lexicographic order, as an (n!, n) int8 array."""
    import itertools
    return np.array(list(itertools.permutations(range(n))), dtype=np.int8)


def rank(p):
    n = len(p)
    return sum(sum(1 for j in range(i + 1, n) if p[j] < p[i]) * math.factorial(n - 1 - i) for i in range(n))


def contains_312(w):
    n = len(w)
    return any(w[i] > w[k] > w[j] for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n))


# --------------------------------------------------------------------------- part B helpers

@njit(cache=True)
def _top_ranks(perms, Lb, Rb, fact):
    """rank of v* = max of W_I v W_J for every v.  Lb[i] = 1 iff s_i in I (values i, i+1), Rb likewise
    for J (positions i, i+1)."""
    N, n = perms.shape
    out = np.empty(N, np.int64)
    v = np.empty(n, np.int64)
    pos = np.empty(n, np.int64)
    for r in range(N):
        for i in range(n):
            v[i] = perms[r, i]
        changed = True
        while changed:
            changed = False
            for i in range(n):
                pos[v[i]] = i
            # left: value i+1 must stand left of i for every s_i in I (bubble within value blocks)
            for sweep in range(n):
                again = False
                for i in range(n - 1):
                    if Lb[i] and pos[i + 1] > pos[i]:
                        a, b = pos[i], pos[i + 1]
                        v[a], v[b] = i + 1, i
                        pos[i], pos[i + 1] = b, a
                        again = changed = True
                if not again:
                    break
            # right: v[j] > v[j+1] for every s_j in J
            for sweep in range(n):
                again = False
                for j in range(n - 1):
                    if Rb[j] and v[j] < v[j + 1]:
                        v[j], v[j + 1] = v[j + 1], v[j]
                        again = changed = True
                if not again:
                    break
        rk = 0
        for i in range(n):
            c = 0
            for j in range(i + 1, n):
                if v[j] < v[i]:
                    c += 1
            rk += c * fact[n - 1 - i]
        out[r] = rk
    return out


@njit(parallel=True, cache=True)
def _weighted_sums(coef, rows, out):
    """out[k] = sum_v coef[v] rows[k, v] mod P (coef, rows in [0, P))."""
    N = coef.shape[0]
    nchunk = 256
    part = np.zeros((nchunk, rows.shape[0]), np.int64)
    for c in prange(nchunk):
        lo = c * N // nchunk
        hi = (c + 1) * N // nchunk
        for k in range(rows.shape[0]):
            s = 0
            for v in range(lo, hi):
                s = (s + coef[v] * np.int64(rows[k, v])) % P
            part[c, k] = s
    for k in range(rows.shape[0]):
        s = 0
        for c in range(nchunk):
            s = (s + part[c, k]) % P
        out[k] = s


def kl_columns(n, ws, perms, fact, batch=20):
    """For each w: coef[i] = array over v of P_{v,w}(q_i) mod P (0 unless v <= w), from Coxeter3."""
    out = []
    for b in range(0, len(ws), batch):
        chunk = ws[b:b + batch]
        words = [reduced_word(np.array(w, dtype=np.int8)) for w in chunk]
        blocks = parse_blocks(run_coxeter3(n, words, 3600), len(chunk))
        for w, block in zip(chunk, blocks):
            I = np.array([1 if w.index(i + 1) < w.index(i) else 0 for i in range(n - 1)], np.int64)
            J = np.array([1 if w[j] > w[j + 1] else 0 for j in range(n - 1)], np.int64)
            top = _top_ranks(perms, I, J, fact)
            ext = {rank(word_to_perm(x, n)): c for x, c in block}
            assert rank(w) in ext and ext[rank(w)] == [1]
            cols = []
            for q in TRACE_QS:
                val = np.zeros(len(perms), np.int64)
                lut = np.zeros(len(perms), np.int64)
                for x, c in ext.items():
                    lut[x] = sum(a * pow(q, e, P) for e, a in enumerate(c)) % P
                val[:] = lut[top]
                cols.append(val)
            out.append(cols)
    return out


def stream_table(zst, ws, sha_file=None):
    """C[:, w, :] for the given w ranks, by one pass over ``zstd -dc`` of the npz (stored zip member C.npy).
    Also returns the SHA-256 of the decompressed npz and whether it matches ``sha_file``."""
    proc = subprocess.Popen(["zstd", "-dc", "-T0", zst], stdout=subprocess.PIPE, bufsize=1 << 24)
    f = proc.stdout
    h = hashlib.sha256()

    def read(k):
        b = f.read(k)
        assert len(b) == k, "unexpected end of stream"
        h.update(b)
        return b

    hdr = read(30)
    assert hdr[:4] == b"PK\x03\x04" and struct.unpack("<H", hdr[8:10])[0] == 0, "expected a stored zip member"
    nlen, xlen = struct.unpack("<HH", hdr[26:30])
    name = read(nlen)
    assert name == b"C.npy", name
    read(xlen)
    magic = read(8)
    assert magic[:6] == b"\x93NUMPY"
    hl = struct.unpack("<H", read(2))[0] if magic[6] == 1 else struct.unpack("<I", read(4))[0]
    meta = eval(read(hl).decode("latin1"))          # numpy's own header format (a dict literal)
    assert meta["descr"] == "<i8" and not meta["fortran_order"]
    L, N, D = meta["shape"]
    want = np.array(sorted(set(ws)), np.int64)
    out = np.zeros((L, len(want), D), np.int64)
    row = D * 8
    per = max(1, (1 << 26) // row)                 # rows per read (~64 MB)
    for l in range(L):
        for a in range(0, N, per):
            b = min(N, a + per)
            buf = np.frombuffer(read((b - a) * row), dtype="<i8").reshape(b - a, D)
            sel = np.flatnonzero((want >= a) & (want < b))
            out[l, sel] = buf[want[sel] - a]
    while True:                                    # the rest (parts.npy, central directory), for the hash
        b = f.read(1 << 24)
        if not b:
            break
        h.update(b)
    proc.wait()
    assert proc.returncode == 0
    digest = h.hexdigest()
    ok = None
    if sha_file:
        ok = open(sha_file).read().split()[0] == digest
    pos = {int(w): k for k, w in enumerate(want)}
    return {w: out[:, pos[w], :] for w in ws}, digest, ok


# --------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--mu", help="space-separated list like '9,1 8,2' (default: the six largest-stabiliser shapes at n = 10)")
    ap.add_argument("--traces", default=None, help="traces npz (default results/kl_graded_n{n}_traces_modp.npz)")
    ap.add_argument("--table", default=None, help="zstd table (default results/kl_graded_n{n}.npz.zst)")
    ap.add_argument("--fresh", action="store_true", help="recompute traces and table in memory (n <= 9) instead")
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skip-b", action="store_true", help="part A only")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    n = a.n
    parts = [tuple(x) for x in partitions(n)]
    if a.mu:
        mus = [tuple(int(x) for x in m.split(",")) for m in a.mu.split()]
    elif n in DEFAULT_MU:
        mus = [tuple(int(x) for x in m.split(",")) for m in DEFAULT_MU[n].split()]
    else:
        mus = [mu for mu in parts if math.factorial(n) // math.prod(math.factorial(b) for b in mu) <= 300]
    out = {"n": n, "q": list(TRACE_QS), "prime": P, "mu": [list(m) for m in mus],
           "dims": [math.factorial(n) // math.prod(math.factorial(b) for b in mu) for mu in mus]}

    C_fresh = None
    if a.fresh:
        from run_kl_graded import traces_mod_p
        from monimm.kl_extremal import ExtremalKL
        from monimm.kl_graded import c_polynomials, hecke_traces
        K = ExtremalKL(n)
        Tr = hecke_traces(K)
        E = traces_mod_p(Tr).astype(np.uint32)
        C_fresh = c_polynomials(K, Tr)
        del Tr
        log(f"S_{n}: traces and table recomputed")
    else:
        z = np.load(a.traces or f"results/kl_graded_n{n}_traces_modp.npz")
        assert int(z["p"]) == P and tuple(z["q"]) == TRACE_QS
        assert [str(p) for p in parts] == list(z["parts"]), "shape order differs"
        E = z["E"]
        log(f"traces loaded: {E.shape}")
    Nm = np.array([[_count(mu, lam) for lam in parts] for mu in mus], np.int64)

    # ---- part A
    eta = {}
    bad_a = {}
    for mu, Nrow in zip(mus, Nm):
        for i, q in enumerate(TRACE_QS):
            t = time.time()
            got = eta_traces_all(n, mu, q, P)
            want = np.zeros(E.shape[2], np.int64)
            for l in range(len(parts)):
                if Nrow[l]:
                    want = (want + int(Nrow[l]) * E[i, l].astype(np.int64)) % P
            bad = int((got != want).sum())
            bad_a[f"{mu} q={q}"] = bad
            eta[(mu, i)] = got
            log(f"A: mu = {mu}, q = {q}: {bad} mismatching v of {len(got)} ({time.time() - t:.0f} s)")
    out["A_mismatching_v"] = bad_a
    out["A_pass"] = all(v == 0 for v in bad_a.values())

    # ---- part B
    if not a.skip_b:
        perms = lex_perms(n)
        fact = np.array([math.factorial(k) for k in range(n + 1)], np.int64)
        rng = np.random.default_rng(a.seed)
        ws = []
        while len(ws) < a.samples:
            w = tuple(int(x) for x in rng.permutation(n))
            if contains_312(w) and w not in ws:
                ws.append(w)
        ranks = [rank(w) for w in ws]
        if C_fresh is not None:
            Cw = {r: C_fresh[:, r, :] for r in ranks}
            digest = ok_hash = None
        else:
            table = a.table or f"results/kl_graded_n{n}.npz.zst"
            log(f"B: streaming {table} for {len(ws)} w")
            sha = table[:-4] + ".sha256" if table.endswith(".zst") else None
            Cw, digest, ok_hash = stream_table(table, ranks, sha if sha and os.path.exists(sha) else None)
            log(f"B: table streamed, sha256 {digest} ({'matches' if ok_hash else 'DOES NOT match'} the stored hash)")
        out["table_sha256"] = digest
        out["table_sha256_matches"] = ok_hash
        log("B: Kazhdan-Lusztig columns from Coxeter3")
        cols = kl_columns(n, ws, perms, fact)
        b1 = b2 = 0
        qpow = [np.array([pow(q, k, P) for k in range(46)], dtype=object) for q in TRACE_QS]
        ETA = np.stack([np.stack([eta[(mu, i)] for mu in mus]) for i in range(len(TRACE_QS))])
        bad_ws = []
        for w, r, wc in zip(ws, ranks, cols):
            C = Cw[r]
            bad = False
            for i in range(len(TRACE_QS)):
                Cq = np.array([int(sum(int(C[l, k]) * qpow[i][k] for k in range(C.shape[1]) if C[l, k]) % P)
                               for l in range(len(parts))], dtype=object)
                s = np.zeros(len(parts), np.int64)
                _weighted_sums(wc[i], E[i], s)
                b1 += int((s.astype(object) != Cq).sum())
                t = np.zeros(len(mus), np.int64)
                _weighted_sums(wc[i], ETA[i], t)
                lhs = [(sum(int(Nm[k, l]) * Cq[l] for l in range(len(parts)))) % P for k in range(len(mus))]
                b2 += sum(int(x != int(y)) for x, y in zip(lhs, t))
                bad |= bool((s.astype(object) != Cq).any()) or any(int(x) != int(y) for x, y in zip(lhs, t))
            if bad:
                bad_ws.append(list(w))
        out["B_samples"] = len(ws)
        out["B_seed"] = a.seed
        out["B_lengths"] = [min(sum(1 for i in range(n) for j in range(i + 1, n) if w[i] > w[j]) for w in ws),
                            max(sum(1 for i in range(n) for j in range(i + 1, n) if w[i] > w[j]) for w in ws)]
        out["B1_mismatching_entries"] = b1
        out["B2_mismatching_entries"] = b2
        out["B_mismatching_w"] = bad_ws[:20]
        out["B_pass"] = b1 == 0 and b2 == 0 and ok_hash is not False
        log(f"B: (B1) {b1} and (B2) {b2} mismatching entries over {len(ws)} w")
    out["seconds"] = round(time.time() - log.__defaults__[0][0], 1)
    print(json.dumps(out, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0 if out["A_pass"] and out.get("B_pass", True) else 1


if __name__ == "__main__":
    sys.exit(main())
