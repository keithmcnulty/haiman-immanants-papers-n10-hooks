"""Irreducible Hecke characters chi^lambda_q(T_v) for every v in S_n, from Hoefsmit's seminormal form.

The mathematics
---------------
H_n(q) has generators T_1, ..., T_{n-1} with (T_s - q)(T_s + 1) = 0 and the braid relations; the index
representation is T_s -> q (the convention of :mod:`monimm.induced_traces` and :mod:`monimm.kl_graded`).
For lambda |- n let Tab(lambda) be the standard Young tableaux of shape lambda, and for an entry x of a
tableau t let c_t(x) = column - row be its content.  Hoefsmit's seminormal form (P. N. Hoefsmit, doctoral
dissertation, Univ. of British Columbia, 1974; see also A. Ram, Proc. London Math. Soc. 75 (1997) 99-133) is
the H_n(q)-module S^lambda with basis {e_t : t in Tab(lambda)} on which s = s_i,
the transposition of the values i and i+1 (here 0-indexed, so the tableau entries i+1, i+2), acts as
follows.  Put d = c_t(i+2) - c_t(i+1), the axial distance, and

    a(d) = (q - 1) q^d / (q^d - 1).

  * i+1, i+2 in the same row of t (then d = 1):     T_s e_t = q e_t;
  * i+1, i+2 in the same column of t (then d = -1): T_s e_t = -e_t;
  * otherwise t' = s_i t (swap the two entries) is again standard, d' = -d, and

        T_s e_t  = a(d) e_t   + e_{t'}                       if d > 0,
        T_s e_t' = a(-d) e_t' + (a(d) a(-d) + q) e_t.

On span{e_t, e_t'} the matrix of T_s has trace a(d) + a(-d) = q - 1 and determinant -q, so it satisfies
the quadratic relation; the braid relations hold by Hoefsmit's theorem, and ``check_relations`` verifies
both (mod p) for every lambda it is asked about.  The denominators q^d - 1 with 1 <= |d| <= n - 1 are
invertible mod the prime p used here (q = 2, 3 have multiplicative order > 9 mod p).  With this
convention S^{(n)} is the index representation (one tableau, all entries in one row), and the character
chi^lambda_q of S^lambda is the one with Frobenius image s_lambda (``tests`` and the drivers confirm this
against the induced modules M^mu, whose characters satisfy eta^mu_q = sum_lambda K_{lambda mu} chi^lambda_q).

Use for check 12 of paper II
-----------------------------
The monomial characters are phi^lambda_q = sum_mu (K^{-1})_{lambda mu} chi^mu_q, i.e.
chi^mu_q = sum_lambda K_{mu lambda} phi^lambda_q with K the (unitriangular) Kostka matrix, s_mu = sum_lambda
K_{mu lambda} m_lambda.  So computing chi^mu_q(T_v) for every v and every mu, and comparing with
sum_lambda K_{mu lambda} phi^lambda_q(T_v) from the Geck-Pfeiffer class polynomials, tests every one of
the 42 trace columns at every v in S_10, with no class polynomial, no character formula and no
double-coset sum.  The induced modules M^mu of :mod:`monimm.induced_traces` reach only the six mu with
dim M^mu <= 252; the irreducible modules have dimension f^lambda <= 768 for n = 10, and the total work is
proportional to sum_lambda (f^lambda)^2 = n!.

The kernel
----------
The tree walk over the left weak order is that of :mod:`monimm.induced_traces` (one sparse left
multiplication per v, a stack of matrices along the current path).  The tree is very unbalanced: cut at
length 8 in S_10, two subtrees hold 403,200 and 362,880 of the 10! nodes, so a uniform cut leaves most
of 192 threads idle.  ``tree_plan`` therefore computes every subtree size and splits adaptively: a node
whose subtree is larger than a threshold becomes a single-node task and its children are examined in
turn; a smaller subtree becomes one task.  The tasks run largest first with dynamic scheduling
(``numba.set_parallel_chunksize(1)``).  The arithmetic, and hence every output, is the same for any plan.  In the seminormal form every row of
T_s has at most two non-zero entries, on the diagonal and at the partner tableau, so a step is

    Y[r, :] = A[i, r] X[r, :] + B[i, r] X[partner[i, r], :]     (mod p),

one reduction per entry.  All values lie in [0, p) with p < 2^31, so each product is < 2^62 and the sum of
two is < 2^63: int64 never overflows.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from monimm.induced_traces import _is_child, _rank, _swap_values, _trace


# --------------------------------------------------------------------------- the seminormal form

def standard_tableaux(shape):
    """All standard Young tableaux of the given shape, each as a tuple (row, col) per entry 0..n-1."""
    n = sum(shape)
    out = []
    rows = [0] * len(shape)                       # current length of each row

    def rec(k, cells):
        if k == n:
            out.append(tuple(cells))
            return
        for r in range(len(shape)):
            c = rows[r]
            if c < shape[r] and (r == 0 or rows[r - 1] > c):
                rows[r] += 1
                cells.append((r, c))
                rec(k + 1, cells)
                cells.pop()
                rows[r] -= 1
    rec(0, [])
    return out


def seminormal_tables(shape, q, p):
    """A, B, partner (each (n-1, f) int64) with Y = T_{s_i} X given row-wise by
    Y[r] = A[i, r] X[r] + B[i, r] X[partner[i, r]]  (partner = r and B = 0 if there is none)."""
    tabs = standard_tableaux(shape)
    n = sum(shape)
    f = len(tabs)
    idx = {t: k for k, t in enumerate(tabs)}
    inv = lambda x: pow(x % p, p - 2, p)
    qp = lambda d: pow(q, d, p) if d >= 0 else pow(inv(q), -d, p)
    a = lambda d: (q - 1) * qp(d) % p * inv(qp(d) - 1) % p
    A = np.zeros((n - 1, f), np.int64)
    B = np.zeros((n - 1, f), np.int64)
    partner = np.tile(np.arange(f, dtype=np.int64), (n - 1, 1))
    for k, t in enumerate(tabs):
        for i in range(n - 1):
            (r1, c1), (r2, c2) = t[i], t[i + 1]
            d = (c2 - r2) - (c1 - r1)
            if r1 == r2:
                A[i, k] = q % p
            elif c1 == c2:
                A[i, k] = p - 1
            else:
                s = list(t)
                s[i], s[i + 1] = s[i + 1], s[i]
                k2 = idx[tuple(s)]
                # column k of T_s: a(d) e_k + (1 if d > 0 else a(d) a(-d) + q) e_{k2};
                # in row form, row k gets a(d) X[k] and row k2 gets that coefficient times X[k]
                A[i, k] = a(d)
                partner[i, k2] = k
                B[i, k2] = 1 if d > 0 else (a(d) * a(-d) + q) % p
    return A, B, partner


def dense_generators(shape, q, p):
    """The matrices of T_1, ..., T_{n-1} on S^lambda (column convention: column k = T_s e_k), mod p."""
    A, B, partner = seminormal_tables(shape, q, p)
    n1, f = A.shape
    mats = []
    for i in range(n1):
        M = np.zeros((f, f), dtype=object)
        for r in range(f):
            M[r, r] = int(A[i, r])
            if B[i, r]:
                M[r, partner[i, r]] = (M[r, partner[i, r]] + int(B[i, r])) % p
        mats.append(M)
    return mats


def check_relations(shape, q, p):
    """Quadratic and braid relations for the seminormal generators mod p (dense; small shapes only)."""
    T = dense_generators(shape, q, p)
    f = T[0].shape[0] if T else 1
    I = np.eye(f, dtype=object)
    mm = lambda X, Y: (X.dot(Y)) % p
    for i, X in enumerate(T):
        if ((mm(X - q * I, X + I)) % p).any():
            return False, f"quadratic relation fails at s_{i}"
    for i in range(len(T)):
        for j in range(len(T)):
            if abs(i - j) == 1 and i < j:
                if (mm(mm(T[i], T[j]), T[i]) != mm(mm(T[j], T[i]), T[j])).any():
                    return False, f"braid relation fails at s_{i}, s_{j}"
            elif abs(i - j) > 1 and (mm(T[i], T[j]) != mm(T[j], T[i])).any():
                return False, f"commutation fails at s_{i}, s_{j}"
    return True, "ok"


# --------------------------------------------------------------------------- the kernel

@njit(cache=True, inline="always")
def _apply_sn(X, Y, i, A, B, partner, p):
    """Y = T_{s_i} X in the seminormal form (rows of X are coordinates, columns are basis images)."""
    m = X.shape[0]
    for r in range(m):
        a = A[i, r]
        b = B[i, r]
        if b == 0:
            for c in range(m):
                Y[r, c] = (a * X[r, c]) % p
        else:
            s = partner[i, r]
            for c in range(m):
                Y[r, c] = (a * X[r, c] + b * X[s, c]) % p


@njit(parallel=True, cache=True)
def _run_sn(words, wlen, expand, n, A, B, partner, p, fact, out):
    """As ``monimm.induced_traces._run``, with the seminormal step."""
    m = A.shape[1]
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
            _apply_sn(X, tmp, j, A, B, partner, p)
            X[:, :] = tmp
        out[_rank(perm, fact)] = _trace(X, p)
        if expand[t] == 0:
            continue
        nxt = np.zeros(D - L + 2, np.int64)
        used = np.zeros(D - L + 2, np.int64)
        h = 0
        while h >= 0:
            j = nxt[h]
            if j == n - 1:
                if h > 0:
                    _swap_values(perm, pos, used[h])
                h -= 1
                continue
            nxt[h] = j + 1
            if pos[j] > pos[j + 1]:
                continue
            _swap_values(perm, pos, j)
            if not _is_child(pos, j):
                _swap_values(perm, pos, j)
                continue
            _apply_sn(stack[h], stack[h + 1], j, A, B, partner, p)
            out[_rank(perm, fact)] = _trace(stack[h + 1], p)
            h += 1
            used[h] = j
            nxt[h] = 0


_PLANS = {}


def tree_plan(n, threshold=None, threads=None):
    """Tasks (words from e, expand flags), largest subtree first.  The parent of v != e is s_i v with i the
    smallest left descent of v (as in monimm.induced_traces), so the children of u are the s_j u > u for
    which j is the smallest left descent of s_j u."""
    import numba
    threads = threads or numba.config.NUMBA_NUM_THREADS
    N = math.factorial(n)
    threshold = threshold or max(64, N // (8 * threads))
    key = (n, threshold)
    if key in _PLANS:
        return _PLANS[key]
    import itertools
    perms = np.array(list(itertools.permutations(range(n))), dtype=np.int64)       # lexicographic order
    fact = np.array([math.factorial(k) for k in range(n + 1)], np.int64)
    pos = np.argsort(perms, axis=1)
    desc = pos[:, 1:] < pos[:, :-1]                    # desc[:, i]: value i+1 stands left of i
    has = desc.any(axis=1)
    i = desc.argmax(axis=1)
    par = perms.copy()                                 # parent: swap the values i, i+1
    r = np.arange(N)
    a, b = pos[r, i], pos[r, i + 1]
    par[r, a], par[r, b] = i + 1, i
    weights = fact[n - 1 - np.arange(n)]
    lehmer = np.zeros((N, n), np.int64)
    for k in range(n):
        lehmer[:, k] = (par[:, k + 1:] < par[:, k:k + 1]).sum(axis=1)
    prank = np.where(has, lehmer @ weights, -1)
    gen = np.where(has, i, -1)
    length = np.zeros(N, np.int64)
    for k in range(n):
        length += (perms[:, k + 1:] < perms[:, k:k + 1]).sum(axis=1)
    size = np.ones(N, np.int64)
    for v in np.argsort(-length, kind="stable"):       # children before parents
        if prank[v] >= 0:
            size[prank[v]] += size[v]
    children = [[] for _ in range(N)]
    for v in np.flatnonzero(has):
        children[prank[v]].append(v)
    tasks = []                                          # (size, word, expand)
    stack = [(0, [])]                                   # rank 0 is e
    while stack:
        v, word = stack.pop()
        if size[v] <= threshold:
            tasks.append((int(size[v]), word, 1))
        else:
            tasks.append((1, word, 0))
            for c in children[v]:
                stack.append((c, word + [int(gen[c])]))
    tasks.sort(key=lambda t: -t[0])
    assert sum(t[0] for t in tasks) == N
    D = max(1, max(len(t[1]) for t in tasks))
    words = np.zeros((len(tasks), D), np.int64)
    wlen = np.zeros(len(tasks), np.int64)
    expand = np.zeros(len(tasks), np.int64)
    for k, (_, w, e) in enumerate(tasks):
        words[k, :len(w)] = w
        wlen[k] = len(w)
        expand[k] = e
    plan = (words, wlen, expand, max(t[0] for t in tasks))
    _PLANS[key] = plan
    return plan


def chi_traces_all(n, shape, q, p, threshold=None, cut=None):
    """out[r] = chi^shape_q(T_v) mod p for v of lexicographic rank r, every v in S_n.
    (``cut`` is accepted for compatibility and ignored: the plan is adaptive, see ``tree_plan``.)"""
    import numba
    shape = tuple(shape)
    if n == 1:
        return np.ones(1, np.int64)
    A, B, partner = seminormal_tables(shape, q, p)
    words, wlen, expand, _ = tree_plan(n, threshold)
    fact = np.array([math.factorial(k) for k in range(n + 1)], np.int64)
    out = np.full(math.factorial(n), -1, np.int64)
    old = numba.get_parallel_chunksize()
    numba.set_parallel_chunksize(1)
    try:
        _run_sn(words, wlen, expand, n, A, B, partner, np.int64(p), fact, out)
    finally:
        numba.set_parallel_chunksize(old)
    assert (out >= 0).all(), "some v was not reached by the tree walk"
    return out
