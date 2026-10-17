#!/usr/bin/env python3
"""Exact checks of the finite-n claims about hook expansions and the shapes (3,2^b,1^c).

    .venv/bin/python check_further_computations.py --nmax 8 --json results/further_computations_n8.json

About 3 minutes and under 4 GB for n <= 8.  These checks support statements in the paper on hooks
(Section "Explicit expansions for hooks": Conjecture 10.1, Table 1, the single-block property and the
inertness property) and in the computational paper (Section "Further computations", subsections
"Hook expansions" and "The shapes (3,2^b,1^c)").  They are evidence for conjectures, not proofs.

Notation
--------
Permutations are 0-indexed one-line tuples p, with p[k] the image of k; the monomial of p is
x^p = prod_k x_{k, p(k)}.  For f : S_n -> Z put d_f = sum_v f(v) x^v.  Its coefficient of the
Kazhdan-Lusztig immanant Imm_w is

    f(C'_w(1)) = sum_{v <= w} P_{v,w}(1) f(v),

computed by :func:`monimm.kl_fast.evaluate_on_kl_basis` from the compiled KL tables of
:class:`monimm.kl_fast.FastKL`.  Statements below are 1-indexed, as in the papers.

The checks
----------
1. Conjecture 10.1 (B_ij).  For a cycle gamma on T subset [n] let

       F_gamma(v) = [v agrees with gamma on T] * sgn(v restricted to the complement of T),

   so that d_{F_gamma} = x_gamma det x[T^c, T^c].  For gamma = (i j), B_ij(w) = F_(ij)(C'_w).  The
   conjecture predicts B_ij for every i < j:

   (1) j - i >= 2, Q = {i-1, i, j, j+1} cap [n]: B_ij(w) = 1 iff w fixes [n] \\ Q and its pattern on Q
       lies in the Bruhat interval [(i j), (min Q  max Q)] of S_|Q|; otherwise 0.
   (2) j = i + 1, Q = {i-1, ..., i+2} cap [n]: B_ij(w) = 1 if w fixes [n] \\ Q and its pattern on Q
       is >= (i i+1), except the longest element when |Q| = 4; 2 on the pattern 3412; 1 on the
       5-cycles (i-2, i, i+2, i-1, i+1), (i-2, i+1, i-1, i+2, i) for i >= 3 and
       (i-1, i+1, i+3, i, i+2), (i-1, i+2, i, i+3, i+1) for i + 3 <= n; 0 elsewhere.

   Every (i, j) and every w in S_n is compared, for every n <= nmax.  The table of the paper
   (interior adjacent case, n = 6, i = 3: 23 elements) is asserted separately.

2. Single blocks.  For a >= 2 and every vertex i, f^(a)_i(C'_w) = 0 unless w moves a single block of
   consecutive letters containing i, where f^(a)_i(v) = (-1)^(a-1) sgn(v) [the cycle of v through i
   has length >= a].  Write w = w_1 + w_2 + ... as a direct sum of irreducible blocks of consecutive
   letters (cut at every m with w([1, m]) = [1, m]); "w moves a single block containing i" means that
   exactly one block has size >= 2 and it contains i.  (The set {k : w(k) != k} need not be an
   interval: 321 fixes 2.)  The values come from :func:`monimm.kl_fast.vertex_hook_table`.

3. Inertness.  For a 2- or 3-cycle gamma on T and a point k not in T with distance >= 2 from T, under
   one of the conditions
       (a) min T < k < max T,   (b) no two elements of T are adjacent,
       (c) gamma is an adjacent transposition and k has distance >= 3 from T,
   every w with F_gamma(C'_w) != 0 fixes k, and deleting k gives the coefficients in S_{n-1}:
   F_gamma(C'_w) = F_gamma'(C'_w') for every w fixing k, where w' and gamma' are w and gamma with k
   deleted and the letters standardised.  Since w -> w' is a bijection from the w fixing k onto
   S_{n-1}, this compares the two coefficient vectors in full.

4. The shapes (3,2^b,1^c).  The number of indecomposable 321-avoiding w in S_n containing both 312
   and 231 (indecomposable: no 1 <= m < n with w([1, m]) = [1, m]).  The paper states 40 for
   (3,2,1) (n = 6) and 427 for (3,2,2,1) (n = 8); n = 7, the shapes (3,2,2) and (3,2,1,1), is
   recorded too.  This is pure enumeration, no KL polynomials.

The script exits with status 1 if any check fails, and the JSON summary records every count.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time

import numpy as np

from monimm.kazhdan_lusztig import bruhat_leq
from monimm.kl_fast import FastKL, evaluate_on_kl_basis, vertex_hook_table


# ---------------------------------------------------------------------------
# Permutation helpers (0-indexed one-line tuples)
# ---------------------------------------------------------------------------

def sgn(p):
    """Sign of a permutation, by counting inversions."""
    return -1 if sum(p[a] > p[b] for a in range(len(p)) for b in range(a + 1, len(p))) % 2 else 1


def std(seq):
    """Standardisation: the permutation of 0..len-1 with the same relative order as seq."""
    order = sorted(seq)
    return tuple(order.index(x) for x in seq)


def pattern(p, Q):
    """The pattern of p on the sorted position set Q (assumes p maps Q to Q)."""
    return std([p[q] for q in Q])


def cycle_perm(n, cyc):
    """The permutation of [0, n) with the single cycle cyc = (c_0 -> c_1 -> ... -> c_0)."""
    p = list(range(n))
    for a, b in zip(cyc, cyc[1:] + cyc[:1]):
        p[a] = b
    return tuple(p)


def transposition(m, a, b):
    p = list(range(m))
    p[a], p[b] = b, a
    return tuple(p)


def delete_point(p, k):
    """p with the fixed point k deleted and the letters standardised (p in S_n -> S_{n-1})."""
    return tuple(x - (x > k) for i, x in enumerate(p) if i != k)


def gamma_function(perms, n, cyc):
    """F_gamma(v) for every v (in the order of perms): [v = gamma on T] sgn(v on T^c)."""
    T = sorted(cyc)
    rest = [k for k in range(n) if k not in T]
    target = {a: b for a, b in zip(cyc, cyc[1:] + cyc[:1])}
    f = np.zeros(len(perms), dtype=np.int64)
    for r, v in enumerate(perms):
        if all(v[a] == b for a, b in target.items()):
            f[r] = sgn(std([v[k] for k in rest]))
    return f


def cycles_on(T):
    """The (|T|-1)! cycles with support T, as tuples starting at min T."""
    first, others = T[0], T[1:]
    return [(first,) + c for c in itertools.permutations(others)]


# ---------------------------------------------------------------------------
# 1. Conjecture 10.1
# ---------------------------------------------------------------------------

def predicted_B(n, i, j, perms):
    """The conjectured B_ij (0-indexed i < j) as {w: value}."""
    Q = [q for q in (i - 1, i, j, j + 1) if 0 <= q < n] if j - i >= 2 else \
        [q for q in range(i - 1, i + 3) if 0 <= q < n]
    m = len(Q)
    lo = transposition(m, Q.index(i), Q.index(j))
    out = {}
    if j - i >= 2:
        hi = transposition(m, 0, m - 1)
        for x in itertools.permutations(range(m)):
            if bruhat_leq(lo, x) and bruhat_leq(x, hi):
                out[x] = 1
    else:
        longest = tuple(range(m - 1, -1, -1))
        for x in itertools.permutations(range(m)):
            if bruhat_leq(lo, x) and not (m == 4 and x == longest):
                out[x] = 2 if x == (2, 3, 0, 1) else 1
    # lift the patterns on Q to S_n (fixing [n] \ Q)
    B = {}
    for x, val in out.items():
        w = list(range(n))
        for a, q in enumerate(Q):
            w[q] = Q[x[a]]
        B[tuple(w)] = val
    if j == i + 1:
        cycles = []
        if i >= 2:                                            # i >= 3 when 1-indexed
            cycles += [(i - 2, i, i + 2, i - 1, i + 1), (i - 2, i + 1, i - 1, i + 2, i)]
        if i + 3 <= n - 1:                                    # i + 3 <= n when 1-indexed
            cycles += [(i - 1, i + 1, i + 3, i, i + 2), (i - 1, i + 2, i, i + 3, i + 1)]
        for c in cycles:
            if all(0 <= x < n for x in c):
                B[cycle_perm(n, list(c))] = 1
    return B


def check_conjecture_B(n, perms, E2):
    """Compare every B_ij with the prediction; E2[(i, j)] is the coefficient vector of F_(ij)."""
    bad = []
    for (i, j), vec in E2.items():
        computed = {perms[r]: int(vec[r]) for r in np.nonzero(vec)[0]}
        if computed != predicted_B(n, i, j, perms):
            bad.append([i + 1, j + 1])
    return bad


def check_table(perms, E2):
    """Table 1 of the hooks paper: B_{3,4} for n = 6 has 23 elements, 18 ones from patterns on Q,
    the value 2 at 3412, and the four 5-cycles 345126, 451236, 145623, 156234."""
    vec = E2[(2, 3)]
    B = {''.join(str(x + 1) for x in perms[r]): int(vec[r]) for r in np.nonzero(vec)[0]}
    cycles = {'345126', '451236', '145623', '156234'}
    ok = (len(B) == 23 and sum(1 for v in B.values() if v == 1) == 22
          and [w for w, v in B.items() if v == 2] == ['145236'] and cycles <= set(B))
    return ok


# ---------------------------------------------------------------------------
# 2. Single blocks
# ---------------------------------------------------------------------------

def moved_block(w):
    """(p, q) if w has exactly one irreducible block [p, q] of size >= 2, else None.

    The blocks are the intervals between consecutive cut points m (w([0, m)) = [0, m))."""
    cuts = [m for m in range(1, len(w) + 1) if max(w[:m]) == m - 1]
    blocks = [(a, b - 1) for a, b in zip([0] + cuts[:-1], cuts) if b - a >= 2]
    return blocks[0] if len(blocks) == 1 else None


def check_single_blocks(n, fast):
    perms = fast.tables.perms
    F = vertex_hook_table(fast)                       # F[w, a-1, i] = f^(a)_i(C'_w)
    blocks = [moved_block(w) for w in perms]
    bad, nonzero = 0, 0
    for r in range(len(perms)):
        for a in range(2, n + 1):
            for i in range(n):
                if F[r, a - 1, i]:
                    nonzero += 1
                    b = blocks[r]
                    if b is None or not (b[0] <= i <= b[1]):
                        bad += 1
    return bad, nonzero


# ---------------------------------------------------------------------------
# 3. Inertness
# ---------------------------------------------------------------------------

def inert_points(n, cyc):
    """The k of the inertness property for the cycle cyc (0-indexed)."""
    T = sorted(cyc)
    adjacent_T = any(b - a == 1 for a, b in zip(T, T[1:]))
    adjacent_transposition = len(T) == 2 and T[1] - T[0] == 1
    out = []
    for k in range(n):
        if k in T:
            continue
        dist = min(abs(k - t) for t in T)
        if dist < 2:
            continue
        if T[0] < k < T[-1] or not adjacent_T or (adjacent_transposition and dist >= 3):
            out.append(k)
    return out


def coefficient_vectors(fast, cycles):
    """{cycle: F_cycle(C'_w) for all w} for a list of cycles on [0, n)."""
    perms = fast.tables.perms
    if not cycles:
        return {}
    fvals = np.stack([gamma_function(perms, fast.n, list(c)) for c in cycles])
    E = evaluate_on_kl_basis(fast, fvals)
    return {c: E[r] for r, c in enumerate(cycles)}


def check_inertness(n, fast, vecs, small_fast, small_vecs):
    """Compare F_gamma(C'_w) on S_n with F_gamma'(C'_w') on S_{n-1}, for every gamma and inert k."""
    perms = fast.tables.perms
    small_rank = small_fast.tables.rank
    tested, bad = 0, []
    for cyc, vec in vecs.items():
        for k in inert_points(n, cyc):
            tested += 1
            small_cyc = tuple(x - (x > k) for x in cyc)
            small_vec = small_vecs[small_cyc]
            ok = True
            for r, w in enumerate(perms):
                if w[k] != k:
                    ok = ok and vec[r] == 0          # the support fixes k
                elif vec[r] != small_vec[small_rank[delete_point(w, k)]]:
                    ok = False
                if not ok:
                    break
            if not ok:
                bad.append({"cycle": [x + 1 for x in cyc], "k": k + 1})
    return tested, bad


# ---------------------------------------------------------------------------
# 4. The shapes (3,2^b,1^c)
# ---------------------------------------------------------------------------

def contains(w, pat):
    m = len(pat)
    return any(std([w[a] for a in pos]) == pat for pos in itertools.combinations(range(len(w)), m))


def indecomposable(w):
    return all(set(w[:m]) != set(range(m)) for m in range(1, len(w)))


def count_onethree(n):
    """Indecomposable 321-avoiding w in S_n containing both 312 and 231."""
    return sum(1 for w in itertools.permutations(range(n))
               if indecomposable(w) and not contains(w, (2, 1, 0))
               and contains(w, (2, 0, 1)) and contains(w, (1, 2, 0)))


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nmax", type=int, default=8)
    ap.add_argument("--json", help="write the summary here")
    args = ap.parse_args()

    t0 = time.time()
    summary = {"nmax": args.nmax, "per_n": {}}
    ok = True
    prev = None                                     # (FastKL, coefficient vectors) for n - 1
    for n in range(2, args.nmax + 1):
        fast = FastKL(n)
        perms = fast.tables.perms
        two = [(i, j) for i in range(n) for j in range(i + 1, n)]
        three = [c for T in itertools.combinations(range(n), 3) for c in cycles_on(list(T))]
        vecs = coefficient_vectors(fast, two + three)
        E2 = {c: vecs[c] for c in two}

        row = {}
        row["B_pairs"] = len(two)
        row["B_mismatches"] = check_conjecture_B(n, perms, E2)
        if n == 6:
            row["table_1_ok"] = check_table(perms, E2)
        row["single_block_violations"], row["single_block_nonzero_values"] = check_single_blocks(n, fast)
        if prev is not None:
            row["inert_cases"], row["inert_failures"] = check_inertness(n, fast, vecs, *prev)
        else:
            row["inert_cases"], row["inert_failures"] = 0, []
        passed = (not row["B_mismatches"] and row.get("table_1_ok", True)
                  and row["single_block_violations"] == 0 and not row["inert_failures"])
        row["passed"] = passed
        ok = ok and passed
        summary["per_n"][n] = row
        print(f"n = {n}: B_ij {len(two)} pairs, {len(row['B_mismatches'])} mismatches"
              + (f", Table 1 {'ok' if row['table_1_ok'] else 'FAILS'}" if n == 6 else "")
              + f"; single block: {row['single_block_nonzero_values']} non-zero values, "
              f"{row['single_block_violations']} violations; inertness: {row['inert_cases']} cases, "
              f"{len(row['inert_failures'])} failures  [{time.time() - t0:.0f} s]", flush=True)
        prev = (fast, vecs)

    counts = {n: count_onethree(n) for n in range(6, min(args.nmax, 8) + 1)}
    summary["onethree_counts"] = counts
    expected = {6: 40, 8: 427}
    summary["onethree_ok"] = all(counts[n] == v for n, v in expected.items() if n in counts)
    ok = ok and summary["onethree_ok"]
    print(f"(3,2^b,1^c): indecomposable 321-avoiding w containing 312 and 231: {counts} "
          f"(paper: 40 for n = 6, 427 for n = 8)  {'ok' if summary['onethree_ok'] else 'MISMATCH'}")

    summary["all_passed"] = ok
    summary["seconds"] = round(time.time() - t0, 1)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=1)
    print("all checks pass" if ok else "SOME CHECKS FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
