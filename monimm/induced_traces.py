"""Traces of every T_v on the induced Hecke modules M^mu, for all v in S_n, at n up to 10.

The mathematics
---------------
For mu |- n let M^mu = H_n (x)_{H_mu} (index) be the module induced from the index representation
T_s -> q of the parabolic subalgebra H_mu (S_mu = S_{mu_1} x S_{mu_2} x ... on consecutive positions).
It has the basis {T_d m : d in D_mu}, D_mu the minimal left coset representatives of S_mu, i.e. the
permutations whose one-line notation increases along each block of positions.  For s = s_i (which swaps
the values i, i+1, 0-indexed) the defining relations alone give

    T_s T_d m = T_{sd} m                     if sd > d and sd is again minimal       ("up"),
              = q T_d m                      if sd > d but sd = d t with t in S_mu   ("fixed"),
              = (q - 1) T_d m + q T_{sd} m   if sd < d                               ("down").

In terms of positions: sd > d iff the value i stands left of i+1 in d, and then sd is minimal unless
i and i+1 lie in the same block (where they are then adjacent and the block's order would break).

The character of M^mu is eta^mu_q, and eta^mu_q = sum_lambda N_{mu lambda} phi^lambda_q with N_{mu lambda}
the number of non-negative integer matrices with row sums mu and column sums lambda (h_mu = sum_lambda
N_{mu lambda} m_lambda).  So eta^mu_q(T_v) is a fixed integer combination of the monomial-character traces
phi^lambda_q(T_v) that the graded engine computes from Geck-Pfeiffer class polynomials
(``run_kl_graded.py --traces-npz``), and comparing the two for every v checks those traces with no class
polynomials, no character formula and no double-coset sums.

The kernel
----------
``check_graded_induced.py`` builds the matrix of every T_v level by level, which needs the matrices of a
whole length level at once: far too much memory at n = 10.  Here we walk a spanning tree of the left weak
order depth first instead.  The parent of v != e is s_i v with i the smallest left descent of v (the
smallest i whose value i+1 stands left of i), so every v is reached exactly once along the path
e -> ... -> v of a reduced word, and T_v = T_{s_i} T_{parent}.  Only the stack of matrices along the
current path is kept (depth <= binom(n, 2) + 1, i.e. 46 matrices of size dim M^mu at n = 10), and each step
is one sparse left multiplication by T_{s_i}, O(dim^2).  The tree is cut at a fixed length; the nodes up to
that length are independent tasks (the path matrix is rebuilt from the identity along the reduced word),
run in parallel by Numba.

All arithmetic is modulo a prime p < 2^31: entries stay in [0, p), so q x + y with q <= 3 stays below 2^33
and int64 never overflows.  The trace of T_v is written at the lexicographic rank of v, the order used by
``monimm.kl_extremal`` and by the dumped traces.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from numba import njit, prange

UP, FIXED, DOWN = 0, 1, 2


def coset_reps(mu):
    """D_mu: one-line notations (0-indexed) increasing on each block of positions, in lexicographic order."""
    n = sum(mu)
    reps = []
    # choose the set of values in each block; the block is that set in increasing order
    def rec(k, left, cur):
        if k == len(mu):
            reps.append(tuple(cur))
            return
        for S in itertools.combinations(left, mu[k]):
            rest = [x for x in left if x not in S]
            rec(k + 1, rest, cur + list(S))
    rec(0, list(range(n)), [])
    reps.sort()
    assert len(reps) == math.factorial(n) // math.prod(math.factorial(b) for b in mu)
    return reps


def module_tables(mu):
    """kind[i, d] in {UP, FIXED, DOWN} and tgt[i, d] = index of s_i d (or d itself if FIXED)."""
    n = sum(mu)
    reps = coset_reps(mu)
    idx = {d: k for k, d in enumerate(reps)}
    block = np.repeat(np.arange(len(mu)), mu)                   # block of each position
    m = len(reps)
    kind = np.empty((n - 1, m), np.int64)
    tgt = np.empty((n - 1, m), np.int64)
    for k, d in enumerate(reps):
        pos = [0] * n
        for j, x in enumerate(d):
            pos[x] = j
        for i in range(n - 1):
            a, b = pos[i], pos[i + 1]
            if a < b and block[a] == block[b]:
                kind[i, k], tgt[i, k] = FIXED, k
            else:
                sd = tuple(i + 1 if x == i else i if x == i + 1 else x for x in d)
                kind[i, k] = UP if a < b else DOWN
                tgt[i, k] = idx[sd]
    return kind, tgt


@njit(cache=True, inline="always")
def _apply(X, Y, i, kind, tgt, q, p):
    """Y = T_{s_i} X (X holds the images of the basis in its columns, so this acts on rows)."""
    m = X.shape[0]
    Y[:, :] = 0
    for d in range(m):
        k = kind[i, d]
        t = tgt[i, d]
        if k == 0:                                   # up: T_s e_d = e_{sd}
            for c in range(m):
                Y[t, c] = (Y[t, c] + X[d, c]) % p
        elif k == 1:                                 # fixed: q e_d
            for c in range(m):
                Y[d, c] = (Y[d, c] + q * X[d, c]) % p
        else:                                        # down: (q - 1) e_d + q e_{sd}
            for c in range(m):
                x = X[d, c]
                Y[d, c] = (Y[d, c] + (q - 1) * x) % p
                Y[t, c] = (Y[t, c] + q * x) % p


@njit(cache=True, inline="always")
def _trace(X, p):
    s = 0
    for d in range(X.shape[0]):
        s += X[d, d]
    return s % p


@njit(cache=True, inline="always")
def _rank(perm, fact):
    """Lexicographic rank of a permutation of 0..n-1."""
    n = perm.shape[0]
    r = 0
    for i in range(n):
        c = 0
        for j in range(i + 1, n):
            if perm[j] < perm[i]:
                c += 1
        r += c * fact[n - 1 - i]
    return r


@njit(cache=True, inline="always")
def _is_child(pos, j):
    """Is s_j u a tree child of u (pos = inverse of u)?  Needs s_j u > u, and j the smallest left
    descent of s_j u.  (Called with s_j already applied to pos; i is a left descent iff pos[i+1] < pos[i].)"""
    if pos[j + 1] > pos[j]:
        return False                                 # after the swap, j must be a descent
    for i in range(j):
        if pos[i + 1] < pos[i]:
            return False
    return True


@njit(cache=True, inline="always")
def _swap_values(perm, pos, j):
    a, b = pos[j], pos[j + 1]
    perm[a], perm[b] = j + 1, j
    pos[j], pos[j + 1] = b, a


@njit(parallel=True, cache=True)
def _run(words, wlen, cut, n, kind, tgt, q, p, fact, out):
    """For each task (a node of length wlen[t] <= cut, reached by the word words[t]): record its trace and,
    if its length is exactly cut, every node of its subtree."""
    m = kind.shape[1]
    D = n * (n - 1) // 2
    for t in prange(words.shape[0]):
        L = wlen[t]
        stack = np.zeros((D - L + 2, m, m), np.int64)
        perm = np.arange(n)
        pos = np.arange(n)
        tmp = np.zeros((m, m), np.int64)
        X = stack[0]
        for d in range(m):
            X[d, d] = 1
        for k in range(L):
            j = words[t, k]
            _swap_values(perm, pos, j)
            _apply(X, tmp, j, kind, tgt, q, p)
            X[:, :] = tmp
        out[_rank(perm, fact)] = _trace(X, p)
        if L != cut:
            continue
        # depth-first search below the task root; nxt[h] = next generator to try at depth h
        nxt = np.zeros(D - L + 2, np.int64)
        used = np.zeros(D - L + 2, np.int64)
        h = 0
        while h >= 0:
            j = nxt[h]
            if j == n - 1:                           # all children done: go back up
                if h > 0:
                    _swap_values(perm, pos, used[h])
                h -= 1
                continue
            nxt[h] = j + 1
            if pos[j] > pos[j + 1]:                  # s_j u < u
                continue
            _swap_values(perm, pos, j)
            if not _is_child(pos, j):
                _swap_values(perm, pos, j)
                continue
            _apply(stack[h], stack[h + 1], j, kind, tgt, q, p)
            out[_rank(perm, fact)] = _trace(stack[h + 1], p)
            h += 1
            used[h] = j
            nxt[h] = 0


def _tree_tasks(n, cut):
    """All tree nodes of length <= cut, as words from e (lists of generators)."""
    tasks = [[]]
    frontier = [([], list(range(n)))]
    for _ in range(cut):
        new = []
        for word, pos in frontier:
            for j in range(n - 1):
                if pos[j] > pos[j + 1]:
                    continue
                q = pos[:]
                q[j], q[j + 1] = q[j + 1], q[j]
                if q[j + 1] > q[j] or any(q[i + 1] < q[i] for i in range(j)):
                    continue
                new.append((word + [j], q))
        frontier = new
        tasks += [w for w, _ in new]
    return tasks


def eta_traces_all(n, mu, q, p, cut=6, seed=0):
    """out[r] = eta^mu_q(T_v) mod p for v of lexicographic rank r, every v in S_n."""
    kind, tgt = module_tables(tuple(mu))
    cut = min(cut, n * (n - 1) // 2)
    tasks = _tree_tasks(n, cut)
    order = np.random.default_rng(seed).permutation(len(tasks))   # spread the large subtrees over threads
    words = np.zeros((len(tasks), max(cut, 1)), np.int64)
    wlen = np.zeros(len(tasks), np.int64)
    for r, k in enumerate(order):
        w = tasks[k]
        words[r, :len(w)] = w
        wlen[r] = len(w)
    fact = np.array([math.factorial(k) for k in range(n + 1)], np.int64)
    out = np.full(math.factorial(n), -1, np.int64)
    _run(words, wlen, cut, n, kind, tgt, np.int64(q), np.int64(p), fact, out)
    assert (out >= 0).all(), "some v was not reached by the tree walk"
    return out
