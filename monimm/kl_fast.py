"""Compact, compiled Kazhdan-Lusztig computation for S_n (n <= 8), and c_{lambda,w}.

This computes the same quantities as :mod:`monimm.kazhdan_lusztig`, but with flat
integer arrays and Numba instead of Python dicts and tuples.  That module needs
about 250 bytes per Bruhat pair, about 45 GB for S_8; this one needs about 8.

Data layout
-----------
Permutations are numbered 0..n!-1 in lexicographic order of their one-line
notation (0-indexed), and are processed in order of increasing length.

* ``start[w], size[w]``: the Bruhat interval {x <= w} is
  ``below[start[w] : start[w] + size[w]]``, sorted by rank.
* ``pid[k]``: for the pair (x, w) at position k of that slice, the id of
  P_{x,w}(q) in the polynomial table ``polys[pid]`` (coefficients of
  1, q, q^2, ...).  Equal polynomials share one id (a hash table on the
  coefficient rows).

Bruhat intervals (lifting property)
-----------------------------------
If s w < w and v = s w, then {x <= w} = {x <= v} union {s x : x <= v}.
(Humphreys, *Reflection groups and Coxeter groups*, 5.9.)

KL recursion
------------
The same as :mod:`monimm.kazhdan_lusztig` (Humphreys 7.11).  For x <= w, with
c(x) = [s x < x],

    P_{x,w} = q^{1-c(x)} P_{sx,v} + q^{c(x)} P_{x,v}
              - sum_{z < v, s z < z} mu(z, v) q^{(l(w)-l(z))/2} P_{x,z}.

We evaluate it by scattering, not by looking up each x.  For each y <= v, the
term P_{y,v} contributes q^{c(y)} P_{y,v} at x = y and also at x = s y, since
1 - c(s y) = c(y).  Each correction z contributes -mu q^{...} P_{y,z} at every
y <= z.  Everything is a sequential pass over sorted arrays.

Checks (in the tests, tests/test_engines.py)
--------------------------------------------
* agreement with :class:`monimm.kazhdan_lusztig.KazhdanLusztig` for n <= 6, and
  with the stored c_{lambda,w} for n <= 7;
* the exact identity sum_w c_{lambda,w} Imm_w = sum_v g_lambda(v) x^v on all n!
  permutation monomials.  It holds only if the P's satisfy the KL inversion
  formula, so it also tests the KL polynomials;
* P_{x,w}(0) = 1, deg P_{x,w} <= (l(w)-l(x)-1)/2, and all coefficients >= 0
  (true for Weyl groups, by Kazhdan-Lusztig / Elias-Williamson).
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations
from math import lcm
from typing import Dict, Tuple

import numpy as np
from numba import njit

from .combinatorics import cycle_type, monomial_class_functions, partitions


# ---------------------------------------------------------------------------
# Permutation tables (plain Python/NumPy; n! <= 40320)
# ---------------------------------------------------------------------------

class PermTables:
    """Ranks, lengths, left multiplication by s_i and by w_0, cycle-type classes."""

    def __init__(self, n: int):
        self.n = n
        self.perms = list(permutations(range(n)))            # lexicographic = rank order
        rank = {p: r for r, p in enumerate(self.perms)}
        N = len(self.perms)
        self.N = N
        self.length = np.array([sum(p[i] > p[j] for i in range(n) for j in range(i + 1, n))
                                for p in self.perms], dtype=np.int32)
        # s_i w swaps the values i and i+1 in one-line notation.
        self.left = np.zeros((max(n - 1, 1), N), dtype=np.int32)
        for i in range(n - 1):
            for r, p in enumerate(self.perms):
                q = tuple(i + 1 if x == i else i if x == i + 1 else x for x in p)
                self.left[i, r] = rank[q]
        self.w0_times = np.array([rank[tuple(n - 1 - x for x in p)] for p in self.perms],
                                 dtype=np.int32)
        self.parts = partitions(n)
        part_index = {lam: k for k, lam in enumerate(self.parts)}
        self.ctype = np.array([part_index[cycle_type(p)] for p in self.perms], dtype=np.int32)
        self.order = np.argsort(self.length, kind="stable").astype(np.int32)
        self.rank = rank


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _build_intervals(order, length, left, n_minus_1, capacity):
    """Bruhat intervals by the lifting property.  Returns (ok, start, size, below, first_descent)."""
    N = order.shape[0]
    start = np.zeros(N, dtype=np.int64)
    size = np.zeros(N, dtype=np.int64)
    first_descent = np.full(N, -1, dtype=np.int32)
    below = np.empty(capacity, dtype=np.int32)
    mark = np.zeros(N, dtype=np.uint8)
    used = 0
    for k in range(N):
        w = order[k]
        if length[w] == 0:
            if used + 1 > capacity:
                return False, start, size, below, first_descent
            start[w] = used
            size[w] = 1
            below[used] = w
            used += 1
            continue
        # First left descent s (s w < w, i.e. length drops).
        s = -1
        for i in range(n_minus_1):
            if length[left[i, w]] < length[w]:
                s = i
                break
        first_descent[w] = s
        v = left[s, w]
        a, m = start[v], size[v]
        # Union of below(v) and s.below(v), via a marker array, then sort.
        count = 0
        for t in range(m):
            x = below[a + t]
            if mark[x] == 0:
                mark[x] = 1
                count += 1
            y = left[s, x]
            if mark[y] == 0:
                mark[y] = 1
                count += 1
        if used + count > capacity:
            return False, start, size, below, first_descent
        out = np.empty(count, dtype=np.int32)
        c = 0
        for t in range(m):
            x = below[a + t]
            if mark[x] == 1:
                out[c] = x
                c += 1
                mark[x] = 2
            y = left[s, x]
            if mark[y] == 1:
                out[c] = y
                c += 1
                mark[y] = 2
        out.sort()
        for t in range(count):
            below[used + t] = out[t]
            mark[out[t]] = 0
        start[w] = used
        size[w] = count
        used += count
    return True, start, size, below[:used], first_descent


@njit(cache=True)
def _hash_row(row):
    h = np.uint64(1469598103934665603)
    for c in row:
        h = (h ^ np.uint64(c & 0xFFFFFFFF)) * np.uint64(1099511628211)
    return h


@njit(cache=True)
def _intern(row, polys, npolys, table, mask):
    """Id of the coefficient row in the polynomial table, inserting it if new."""
    h = _hash_row(row)
    slot = np.int64(h & np.uint64(mask))
    D = row.shape[0]
    while True:
        idx = table[slot]
        if idx == -1:
            pid = npolys[0]
            if pid >= polys.shape[0]:
                return -1
            for d in range(D):
                polys[pid, d] = row[d]
            npolys[0] = pid + 1
            table[slot] = pid
            return pid
        same = True
        for d in range(D):
            if polys[idx, d] != row[d]:
                same = False
                break
        if same:
            return idx
        slot = (slot + 1) & mask


@njit(cache=True)
def _kl(order, length, left, first_descent, start, size, below, D, max_polys, table_bits):
    """KL polynomials for all pairs.  Returns (status, pid, polys, npolys).

    status: 0 ok, 1 polynomial table full, 2 degree bound exceeded.
    """
    N = order.shape[0]
    total = below.shape[0]
    pid = np.empty(total, dtype=np.int32)
    polys = np.zeros((max_polys, D), dtype=np.int32)
    npolys = np.zeros(1, dtype=np.int64)
    table = np.full(1 << table_bits, -1, dtype=np.int32)
    mask = (1 << table_bits) - 1
    pos = np.full(N, -1, dtype=np.int64)            # rank -> position in below(w)
    acc = np.zeros((N, D), dtype=np.int64)
    row = np.zeros(D, dtype=np.int64)
    zs = np.empty(N, dtype=np.int32)
    ms = np.empty(N, dtype=np.int64)
    one = np.zeros(D, dtype=np.int64)
    one[0] = 1
    pid_one = _intern(one, polys, npolys, table, mask)
    for k in range(N):
        w = order[k]
        a, m = start[w], size[w]
        if length[w] == 0:
            pid[a] = pid_one
            continue
        s = first_descent[w]
        v = left[s, w]
        av, mv = start[v], size[v]
        for t in range(m):
            pos[below[a + t]] = t
        # q^{c(y)} P_{y,v} at x = y and x = s y.
        for t in range(mv):
            y = below[av + t]
            sh = 1 if length[left[s, y]] < length[y] else 0
            p = pid[av + t]
            py, psy = pos[y], pos[left[s, y]]
            for d in range(D - sh):
                c = polys[p, d]
                if c != 0:
                    acc[py, d + sh] += c
                    acc[psy, d + sh] += c
        # Corrections: z < v, s z < z, mu(z, v) != 0.
        nz = 0
        for t in range(mv):
            z = below[av + t]
            if z == v or length[left[s, z]] > length[z]:
                continue
            diff = length[v] - length[z]
            if diff % 2 == 0:
                continue
            e = (diff - 1) // 2
            mu = polys[pid[av + t], e] if e < D else 0
            if mu != 0:
                zs[nz] = z
                ms[nz] = mu
                nz += 1
        for j in range(nz):
            z = zs[j]
            mu = ms[j]
            shift = (length[w] - length[z]) // 2
            az, mz = start[z], size[z]
            for t in range(mz):
                y = below[az + t]
                p = pid[az + t]
                py = pos[y]
                for d in range(D - shift):
                    c = polys[p, d]
                    if c != 0:
                        acc[py, d + shift] -= mu * c
        # Intern, check the degree bound, reset.
        for t in range(m):
            x = below[a + t]
            bound = (length[w] - length[x] - 1) // 2 if x != w else 0
            for d in range(D):
                row[d] = acc[t, d]
                acc[t, d] = 0
                if d > bound and row[d] != 0:
                    return 2, pid, polys, npolys[0]
            if x == w:
                row[0] = 1
            r = _intern(row, polys, npolys, table, mask)
            if r < 0:
                return 1, pid, polys, npolys[0]
            pid[a + t] = r
            pos[x] = -1
    return 0, pid, polys, npolys[0]


@njit(cache=True)
def _c_coefficients(start, size, below, pid, values_at_one, ctype, g_scaled):
    """C[l, w] = sum_{x <= w} P_{x,w}(1) G[l, ctype[x]], with G = g scaled to integers."""
    L = g_scaled.shape[0]
    N = start.shape[0]
    out = np.zeros((L, N), dtype=np.int64)
    for w in range(N):
        a, m = start[w], size[w]
        for t in range(m):
            val = values_at_one[pid[a + t]]
            cls = ctype[below[a + t]]
            for l in range(L):
                out[l, w] += val * g_scaled[l, cls]
    return out


@njit(cache=True)
def _identity_check(start, size, below, pid, values_at_one, length, w0_times, C):
    """sum_w C[l,w] Imm_w on monomials: returns T[l, v], the coefficient of x^v.

    Imm_w = sum_{v >= w} (-1)^{l(v)-l(w)} P_{w0 v, w0 w}(1) x^v.  With x = w0 v,
    y = w0 w this is a sum over pairs x <= y, as stored."""
    L = C.shape[0]
    N = start.shape[0]
    T = np.zeros((L, N), dtype=np.int64)
    for y in range(N):
        w = w0_times[y]
        a, m = start[y], size[y]
        for t in range(m):
            x = below[a + t]
            v = w0_times[x]
            val = values_at_one[pid[a + t]]
            if (length[y] - length[x]) % 2 == 1:
                val = -val
            for l in range(L):
                T[l, v] += val * C[l, w]
    return T


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class FastKL:
    """All KL polynomials of S_n in compact form, plus c_{lambda,w}.

    Attributes after construction: ``tables`` (PermTables), ``start``, ``size``,
    ``below``, ``pid``, ``polys`` (distinct polynomials, int32 rows), ``npairs``.
    """

    def __init__(self, n: int, capacity: int = None, max_polys: int = 1 << 20,
                 table_bits: int = 22):
        self.n = n
        t = self.tables = PermTables(n)
        nm1 = max(n - 1, 1) if n > 1 else 0
        capacity = capacity or {1: 1, 2: 3, 3: 19, 4: 213, 5: 3781, 6: 98407,
                                7: 3550919}.get(n, 180_000_000)
        ok, self.start, self.size, self.below, self.first_descent = _build_intervals(
            t.order, t.length, t.left, nm1, capacity)
        if not ok:
            raise MemoryError("Bruhat interval capacity exceeded; pass a larger capacity")
        self.npairs = int(self.below.shape[0])
        D = (n * (n - 1) // 2) // 2 + 2          # max degree of P plus headroom
        status, self.pid, polys, npolys = _kl(t.order, t.length, t.left, self.first_descent,
                                               self.start, self.size, self.below, D,
                                               max_polys, table_bits)
        if status == 1:
            raise MemoryError("polynomial table full; raise max_polys")
        if status == 2:
            raise ArithmeticError("degree bound violated: the recursion is wrong")
        self.polys = polys[:npolys].copy()
        self.values_at_one = self.polys.sum(axis=1).astype(np.int64)

    def p(self, x: Tuple[int, ...], w: Tuple[int, ...]) -> Tuple[int, ...]:
        """P_{x,w} as a coefficient tuple (no trailing zeros), zero tuple if x !<= w."""
        rx, rw = self.tables.rank[x], self.tables.rank[w]
        a, m = self.start[rw], self.size[rw]
        k = np.searchsorted(self.below[a:a + m], rx)
        if k >= m or self.below[a + k] != rx:
            return ()
        row = [int(c) for c in self.polys[self.pid[a + k]]]
        while row and row[-1] == 0:
            row.pop()
        return tuple(row)

    def c_coefficients(self) -> Dict[Tuple[int, ...], Dict[Tuple[int, ...], Fraction]]:
        """c_{lambda,w} for all lambda, w, after the exact identity check on all monomials."""
        t = self.tables
        parts, g = monomial_class_functions(self.n)
        assert tuple(parts) == tuple(t.parts)
        scale = lcm(*[x.denominator for row in g for x in row])
        g_scaled = np.array([[int(x * scale) for x in row] for row in g], dtype=np.int64)
        C = _c_coefficients(self.start, self.size, self.below, self.pid, self.values_at_one,
                            t.ctype, g_scaled)
        # int64 safety for the identity check: |T| <= max|C| * max P(1) * max interval size.
        bound = (int(np.abs(C).max()) * int(self.values_at_one.max()) * int(self.size.max()))
        if bound >= 2 ** 62:
            raise OverflowError("identity check could overflow int64")
        # Identity: sum_w c_{l,w} Imm_w has coefficient g_l(v) on x^v, for every v.
        T = _identity_check(self.start, self.size, self.below, self.pid, self.values_at_one,
                            t.length, t.w0_times, C)
        expected = g_scaled[:, t.ctype]
        if not np.array_equal(T, expected):
            bad = int(np.count_nonzero(T != expected))
            raise ArithmeticError(f"KL expansion identity fails at {bad} (lambda, v) entries")
        self.scale = scale
        self.C_scaled = C
        return {lam: {t.perms[w]: Fraction(int(C[l, w]), scale) for w in range(t.N)}
                for l, lam in enumerate(parts)}

    def sanity(self) -> Dict[str, bool]:
        """P(0) = 1 on every pair and all coefficients >= 0."""
        return {"constant_term_one": bool(np.all(self.polys[:, 0] == 1)),
                "nonnegative_coefficients": bool(np.all(self.polys >= 0))}


# ---------------------------------------------------------------------------
# Per-vertex hook sums (Stembridge's hook formula; Section 4 of the hooks paper)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _vertex_cycle_sums(start, size, below, pid, values_at_one, length, cyc):
    """H[w, i, L] = sum_{x <= w, l_i(x) = L} sgn(x) P_{x,w}(1)."""
    N, n = cyc.shape
    H = np.zeros((N, n, n + 1), dtype=np.int64)
    for w in range(N):
        a, m = start[w], size[w]
        for t in range(m):
            x = below[a + t]
            val = values_at_one[pid[a + t]]
            if length[x] % 2 == 1:
                val = -val
            for i in range(n):
                H[w, i, cyc[x, i]] += val
    return H


def vertex_hook_table(fast: "FastKL") -> np.ndarray:
    """F[w, a-1, i] = f^{(a)}_i(C'_w(1)) = (-1)^{a-1} sum_{L >= a} H[w, i, L] (int64 array).

    Same values as :func:`monimm.hook_kl.vertex_hook_values`, in compact form."""
    t = fast.tables
    n = t.n
    cyc = np.zeros((t.N, n), dtype=np.int8)
    for r, p in enumerate(t.perms):
        for i in range(n):
            j, size = p[i], 1
            while j != i:
                j, size = p[j], size + 1
            cyc[r, i] = size
    H = _vertex_cycle_sums(fast.start, fast.size, fast.below, fast.pid, fast.values_at_one,
                           t.length, cyc)
    tails = np.cumsum(H[:, :, ::-1], axis=2)[:, :, ::-1]      # tails[w, i, L] = sum_{L' >= L}
    F = np.empty((t.N, n, n), dtype=np.int64)
    for a in range(1, n + 1):
        F[:, a - 1, :] = (-1) ** (a - 1) * tails[:, :, a]
    return F


@njit(cache=True)
def _evaluate_on_kl_basis(start, size, below, pid, values_at_one, fvals):
    """E[k, w] = f_k(C'_w(1)) = sum_{x <= w} P_{x,w}(1) f_k(x) for functions f_k : S_n -> Z."""
    K, N = fvals.shape
    E = np.zeros((K, N), dtype=np.int64)
    for w in range(N):
        a, m = start[w], size[w]
        for t in range(m):
            x = below[a + t]
            val = values_at_one[pid[a + t]]
            for k in range(K):
                if fvals[k, x] != 0:
                    E[k, w] += val * fvals[k, x]
    return E


def evaluate_on_kl_basis(fast: "FastKL", fvals: np.ndarray) -> np.ndarray:
    """KL coefficients of the polynomials sum_v f_k(v) x^v: row k is f_k(C'_w(1)) for all w."""
    return _evaluate_on_kl_basis(fast.start, fast.size, fast.below, fast.pid,
                                 fast.values_at_one, np.ascontiguousarray(fvals, dtype=np.int64))
