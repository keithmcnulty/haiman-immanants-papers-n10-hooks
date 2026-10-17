#!/usr/bin/env python3
"""Check 12 of paper/haiman_n10.tex, beyond the test suite: Haiman's polynomials against induced Hecke modules.

The mathematics
---------------
For mu |- n let M^mu = H_n (x)_{H_mu} (index) be the module induced from the index representation
T_s -> q of the parabolic subalgebra H_mu.  It has the basis {T_d m : d a minimal left coset
representative of S_mu} and T_s acts by the defining relations alone:

    T_s T_d m = T_{sd} m                     if sd > d and sd is again minimal,
              = q T_d m                      if sd > d but sd = d t with t in S_mu,
              = (q - 1) T_d m + q T_{sd} m   if sd < d.

Its character is eta^mu_q, and phi^lambda_q = sum_mu (N^{-1})_{lambda mu} eta^mu_q, where N_{mu lambda} is
the number of non-negative integer matrices with row sums mu and column sums lambda (h_mu = sum_lambda
N_{mu lambda} m_lambda).  With the Kazhdan-Lusztig polynomials from the standard recursion
(:class:`monimm.kazhdan_lusztig.KazhdanLusztig`, separate code from the extremal-pair engine), this gives

    c_{lambda,w}(q) = sum_{v <= w} P_{v,w}(q) phi^lambda_q(T_v)

with no class polynomials, no character formula and no double-coset sums.  We evaluate it at q = 2, 3
and compare with :func:`monimm.kl_graded.c_polynomials` evaluated at the same q.

``tests/test_kl_graded.py`` does this exactly over Z for n <= 4.  For n = 6 the module M^{1^6} has
dimension 720, so we work modulo primes p < 2^20 with floating-point matrix products: every entry is
reduced into [0, p) and an inner product of length <= 720 stays below 720 p^2 < 2^53, so the products are
exact.  Agreement modulo two primes for every (lambda, w) is the check.

Usage
-----
    .venv/bin/python check_graded_induced.py --n 6 --json results/graded_induced_n6.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time

import numpy as np
from sympy import Matrix

from monimm.combinatorics import partitions
from monimm.kazhdan_lusztig import KazhdanLusztig, length
from monimm.kl_extremal import ExtremalKL
from monimm.kl_graded import c_polynomials

PRIMES = (1048573, 1048571)       # the two largest primes below 2^20


def _lmul(i, p):
    """s_i p: swap the values i, i+1 (0-indexed)."""
    return tuple(i + 1 if x == i else i if x == i + 1 else x for x in p)


def _count(rows, cols):
    """Number of non-negative integer matrices with the given row and column sums."""
    if not rows:
        return 1 if all(c == 0 for c in cols) else 0
    r, rest = rows[0], rows[1:]
    tot = 0

    def rec(j, left, cur):
        nonlocal tot
        if j == len(cols):
            if left == 0:
                tot += _count(rest, tuple(c - x for c, x in zip(cols, cur)))
            return
        for x in range(min(left, cols[j]) + 1):
            rec(j + 1, left - x, cur + [x])

    rec(0, r, [])
    return tot


def eta_traces(n, parts, perms, q, p):
    """E[mu, v] = eta^mu_q(T_v) mod p, from the induced modules."""
    E = np.zeros((len(parts), len(perms)), dtype=np.int64)
    by_len = sorted(perms, key=length)
    for a, mu in enumerate(parts):
        inner, pos = set(), 0
        for b in mu:
            inner |= set(range(pos, pos + b - 1))
            pos += b
        Dm = [d for d in perms if all(d[i] < d[i + 1] for i in inner)]   # minimal left coset reps
        idx = {d: k for k, d in enumerate(Dm)}
        m = len(Dm)
        Ms = []
        for i in range(n - 1):
            M = np.zeros((m, m))
            for d in Dm:
                sd = _lmul(i, d)
                if length(sd) > length(d):
                    if sd in idx:
                        M[idx[sd], idx[d]] += 1
                    else:
                        M[idx[d], idx[d]] += q
                else:
                    M[idx[d], idx[d]] += q - 1
                    M[idx[sd], idx[d]] += q
            Ms.append(np.mod(M, p))
        # matrices of T_v, one length level at a time (only level l - 1 is needed for level l)
        rank = {v: k for k, v in enumerate(perms)}
        prev = {tuple(range(n)): np.identity(m)}
        E[a, rank[tuple(range(n))]] = m % p
        for lv in range(1, n * (n - 1) // 2 + 1):
            cur = {}
            for v in by_len:
                if length(v) != lv:
                    continue
                i = next(i for i in range(n - 1) if length(_lmul(i, v)) < lv)
                cur[v] = np.mod(Ms[i] @ prev[_lmul(i, v)], p)
                E[a, rank[v]] = int(np.trace(cur[v])) % p
            prev = cur
    return E


def run(n: int, log=print) -> dict:
    t0 = time.time()
    K = ExtremalKL(n)
    T = K.tables
    C = c_polynomials(K)                                  # C[lambda, w, k], the engine under test
    parts = [tuple(x) for x in T.parts]
    assert parts == [tuple(x) for x in partitions(n)]
    perms = [tuple(int(x) for x in T.perms[r]) for r in range(T.N)]
    kl = KazhdanLusztig(n)
    rank = {v: k for k, v in enumerate(perms)}
    log(f"S_{n}: engines built ({time.time() - t0:.0f} s)")
    Nm = Matrix([[_count(mu, lam) for lam in parts] for mu in parts])
    Ninv = Nm.inv()
    assert all(x.is_integer for x in Ninv)
    Ninv = np.array(Ninv.tolist(), dtype=np.int64)       # m_lambda = sum_mu Ninv[lambda, mu] h_mu
    out = {"n": n, "q": [2, 3], "primes": list(PRIMES), "shapes": len(parts), "w": len(perms), "mismatches": 0}
    for q in (2, 3):
        for p in PRIMES:
            E = eta_traces(n, parts, perms, q, p)
            Phi = np.mod(Ninv % p @ E, p)                 # Phi[lambda, v] = phi^lambda_q(T_v) mod p
            # P[v, w] = P_{v,w}(q) mod p
            Pq = np.zeros((len(perms), len(perms)), dtype=np.int64)
            for j, w in enumerate(perms):
                for x in kl.below[w]:
                    c = kl.p(x, w)
                    if c:
                        Pq[rank[x], j] = sum(int(a) * pow(q, e, p) for e, a in enumerate(c)) % p
            direct = (Phi.astype(object) @ Pq.astype(object)) % p   # direct[lambda, w], exact mod p
            qp = np.array([pow(q, k, p) for k in range(C.shape[2])], dtype=object)
            engine = (C.astype(object) @ qp) % p                # engine[lambda, w]
            bad = int((engine != direct).sum())
            out["mismatches"] += bad
            log(f"  q = {q}, p = {p}: mismatching (lambda, w): {bad} ({time.time() - t0:.0f} s)")
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    out = run(a.n)
    print(json.dumps(out, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0 if out["mismatches"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
