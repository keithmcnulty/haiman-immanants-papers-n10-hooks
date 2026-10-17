"""Parallel, memory-lean Kazhdan-Lusztig computation for S_9 (and a cross-check for n <= 8).

Same mathematics as :mod:`monimm.kl_fast` (see its docstring): Bruhat intervals by
the lifting property, the KL recursion evaluated by scattering, interned
polynomials, then c_{lambda,w} and the exact identity check.  What changes is the
engineering needed for S_9, which has 10,501,351,657 Bruhat pairs:

* Permutation tables (ranks via Lehmer codes, lengths, s_i w, w s_i, w_0 w, cycle
  types) are built in compiled code.
* Interval sizes come from a level-by-level bitmap pass, so the interval array
  (int32) and the polynomial-id array (uint32) are allocated once, with exact sizes.
* Every length level is processed in parallel.  A permutation depends only on
  shorter ones, so the elements of one level are independent.  The work of a level
  is split into chunks with private scratch buffers.
* Interning without locks: during a level the global polynomial table is read-only.
  A chunk gives each polynomial not yet in the table a temporary id
  (FLAG | chunk << 24 | local index).  After the level the new polynomials are merged
  sequentially and the temporary ids in that level are patched.
* c_{lambda,w} is computed in parallel over w.  The identity check accumulates into
  private per-chunk arrays that are summed afterwards.

Memory for n = 9: about 42 GB of intervals and 42 GB of polynomial ids, plus small
tables.  Run it on a machine with at least 128 GB (256 GB is comfortable).
"""

from __future__ import annotations

import math
import time
from typing import Dict

import numpy as np
from numba import njit, prange, get_num_threads

from .combinatorics import monomial_class_functions, partitions

FLAG = np.uint32(1 << 31)


# ---------------------------------------------------------------------------
# Permutation tables
# ---------------------------------------------------------------------------

@njit(cache=True)
def _all_perms(n):
    N = 1
    for k in range(2, n + 1):
        N *= k
    out = np.empty((N, n), dtype=np.int8)
    p = np.arange(n).astype(np.int8)
    for r in range(N):
        out[r] = p
        # next permutation in lexicographic order
        i = n - 2
        while i >= 0 and p[i] > p[i + 1]:
            i -= 1
        if i < 0:
            break
        j = n - 1
        while p[j] < p[i]:
            j -= 1
        p[i], p[j] = p[j], p[i]
        p[i + 1:] = p[i + 1:][::-1].copy()
    return out


@njit(cache=True)
def _rank(p, fact):
    n = p.shape[0]
    r = 0
    for i in range(n):
        c = 0
        for j in range(i + 1, n):
            if p[j] < p[i]:
                c += 1
        r += c * fact[n - 1 - i]
    return r


@njit(parallel=True, cache=True)
def _tables(perms, fact):
    N, n = perms.shape
    length = np.zeros(N, dtype=np.int32)
    left = np.zeros((max(n - 1, 1), N), dtype=np.int32)
    right = np.zeros((max(n - 1, 1), N), dtype=np.int32)
    w0t = np.zeros(N, dtype=np.int32)
    code = np.zeros(N, dtype=np.int64)          # cycle type encoded in base n+1, sorted desc
    for r in prange(N):
        p = perms[r]
        inv = 0
        for i in range(n):
            for j in range(i + 1, n):
                if p[i] > p[j]:
                    inv += 1
        length[r] = inv
        q = p.copy()
        for i in range(n - 1):
            # s_i p: swap values i and i+1
            for a in range(n):
                if p[a] == i:
                    q[a] = i + 1
                elif p[a] == i + 1:
                    q[a] = i
                else:
                    q[a] = p[a]
            left[i, r] = _rank(q, fact)
            # p s_i: swap positions i and i+1
            for a in range(n):
                q[a] = p[a]
            q[i], q[i + 1] = p[i + 1], p[i]
            right[i, r] = _rank(q, fact)
        for a in range(n):
            q[a] = n - 1 - p[a]
        w0t[r] = _rank(q, fact)
        seen = np.zeros(n, dtype=np.uint8)
        lens = np.zeros(n, dtype=np.int64)
        m = 0
        for a in range(n):
            if seen[a] == 0:
                L = 0
                b = a
                while seen[b] == 0:
                    seen[b] = 1
                    b = p[b]
                    L += 1
                lens[m] = L
                m += 1
        srt = np.sort(lens[:m])[::-1]
        c = 0
        for k in range(m):
            c = c * (n + 1) + srt[k]
        code[r] = c
    return length, left, right, w0t, code


class BigTables:
    """Tables for S_n; permutations in lexicographic order of one-line notation (0-indexed)."""

    def __init__(self, n: int):
        self.n = n
        self.perms = _all_perms(n)
        self.N = self.perms.shape[0]
        fact = np.array([math.factorial(k) for k in range(n + 1)], dtype=np.int64)
        self.fact = fact
        (self.length, self.left, self.right, self.w0_times, code) = _tables(self.perms, fact)
        self.parts = partitions(n)

        def enc(lam):
            c = 0
            for x in lam:
                c = c * (n + 1) + x
            return c
        index = {enc(lam): k for k, lam in enumerate(self.parts)}
        self.ctype = np.array([index[int(c)] for c in code], dtype=np.int32)
        self.order = np.argsort(self.length, kind="stable").astype(np.int64)
        maxlen = n * (n - 1) // 2
        counts = np.bincount(self.length, minlength=maxlen + 1)
        self.level_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    def rank(self, perm) -> int:
        return int(_rank(np.array(perm, dtype=np.int8), self.fact))


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------

@njit(cache=True)
def _first_descent(left, length, w, n1):
    for i in range(n1):
        if length[left[i, w]] < length[w]:
            return i
    return -1


@njit(parallel=True, cache=True)
def _level_bitmaps(prev_bits, prev_index, ws, left, length, n1, words):
    m = ws.shape[0]
    out = np.zeros((m, words), dtype=np.uint64)
    sizes = np.zeros(m, dtype=np.int64)
    for k in prange(m):
        w = ws[k]
        s = _first_descent(left, length, w, n1)
        src = prev_bits[prev_index[left[s, w]]]
        row = out[k]
        for b in range(words):
            word = src[b]
            while word:
                low = word & (~word + np.uint64(1))
                bit = 0
                t = low
                while t > np.uint64(1):
                    t >>= np.uint64(1)
                    bit += 1
                x = b * 64 + bit
                row[b] |= low
                y = left[s, x]
                row[y >> 6] |= np.uint64(1) << np.uint64(y & 63)
                word ^= low
        c = 0
        for b in range(words):
            x = row[b]
            while x:
                x &= x - np.uint64(1)
                c += 1
        sizes[k] = c
    return out, sizes


def interval_sizes(T: BigTables) -> np.ndarray:
    """size[w] = #{x <= w}, by levels with bitmaps (two levels in memory at a time)."""
    N, n = T.N, T.n
    words = (N + 63) // 64
    size = np.zeros(N, dtype=np.int64)
    index = -np.ones(N, dtype=np.int64)
    e = int(T.order[0])
    prev = np.zeros((1, words), dtype=np.uint64)
    prev[0, e >> 6] = np.uint64(1) << np.uint64(e & 63)
    index[e] = 0
    size[e] = 1
    for l in range(1, len(T.level_start) - 1):
        ws = T.order[T.level_start[l]:T.level_start[l + 1]]
        out, sizes = _level_bitmaps(prev, index, ws, T.left, T.length, n - 1, words)
        index[:] = -1
        index[ws] = np.arange(len(ws))
        size[ws] = sizes
        prev = out
    return size


@njit(parallel=True, cache=True)
def _fill_level(ws, left, length, n1, start, size, below, N, nchunks):
    m = ws.shape[0]
    per = (m + nchunks - 1) // nchunks
    for c in prange(nchunks):
        lo = c * per
        hi = min(m, lo + per)
        if lo >= hi:
            continue
        mark = np.zeros(N, dtype=np.uint8)
        for k in range(lo, hi):
            w = ws[k]
            s = _first_descent(left, length, w, n1)
            v = left[s, w]
            a, mv = start[v], size[v]
            tmp = np.empty(size[w], dtype=np.int32)
            cnt = 0
            for t in range(mv):
                x = below[a + t]
                if mark[x] == 0:
                    mark[x] = 1
                    tmp[cnt] = x
                    cnt += 1
                y = left[s, x]
                if mark[y] == 0:
                    mark[y] = 1
                    tmp[cnt] = y
                    cnt += 1
            tmp = np.sort(tmp[:cnt])
            b0 = start[w]
            for t in range(cnt):
                below[b0 + t] = tmp[t]
                mark[tmp[t]] = 0


# ---------------------------------------------------------------------------
# KL recursion, one level at a time
# ---------------------------------------------------------------------------

@njit(cache=True)
def _hash(row):
    h = np.uint64(1469598103934665603)
    for c in row:
        h = (h ^ np.uint64(c & 0xFFFFFFFF)) * np.uint64(1099511628211)
    return h


@njit(cache=True)
def _lookup(row, polys, table, mask):
    """Global id of row, or -1 (read-only)."""
    slot = np.int64(_hash(row) & np.uint64(mask))
    D = row.shape[0]
    while True:
        idx = table[slot]
        if idx == -1:
            return -1
        same = True
        for d in range(D):
            if polys[idx, d] != row[d]:
                same = False
                break
        if same:
            return idx
        slot = (slot + 1) & mask


@njit(cache=True)
def _insert(row, polys, npolys, table, mask):
    idx = _lookup(row, polys, table, mask)
    if idx >= 0:
        return idx
    pid = npolys[0]
    if pid >= polys.shape[0]:
        return -1
    for d in range(row.shape[0]):
        polys[pid, d] = row[d]
    npolys[0] = pid + 1
    slot = np.int64(_hash(row) & np.uint64(mask))
    while table[slot] != -1:
        slot = (slot + 1) & mask
    table[slot] = pid
    return pid


@njit(parallel=True, cache=True)
def _kl_level(ws, left, length, n1, start, size, below, pid, polys, table, mask,
              D, nchunks, cap, new_rows, new_count, err):
    m = ws.shape[0]
    N = left.shape[1]
    per = (m + nchunks - 1) // nchunks
    for c in prange(nchunks):
        lo = c * per
        hi = min(m, lo + per)
        if lo >= hi:
            continue
        pos = np.full(N, -1, dtype=np.int64)
        maxm = 0
        for k in range(lo, hi):
            if size[ws[k]] > maxm:
                maxm = size[ws[k]]
        acc = np.zeros((maxm, D), dtype=np.int64)
        row = np.zeros(D, dtype=np.int64)
        lmask = 4095
        while (lmask + 1) < 4 * cap:
            lmask = lmask * 2 + 1
        ltable = np.full(lmask + 1, -1, dtype=np.int64)
        zs = np.empty(N, dtype=np.int64)
        ms = np.empty(N, dtype=np.int64)
        for k in range(lo, hi):
            w = ws[k]
            a, mw = start[w], size[w]
            s = _first_descent(left, length, w, n1)
            v = left[s, w]
            av, mv = start[v], size[v]
            for t in range(mw):
                pos[below[a + t]] = t
            for t in range(mv):
                y = below[av + t]
                sy = left[s, y]
                sh = 1 if length[sy] < length[y] else 0
                p = pid[av + t]
                py, psy = pos[y], pos[sy]
                for d in range(D - sh):
                    cc = polys[p, d]
                    if cc != 0:
                        acc[py, d + sh] += cc
                        acc[psy, d + sh] += cc
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
                        cc = polys[p, d]
                        if cc != 0:
                            acc[py, d + shift] -= mu * cc
            for t in range(mw):
                x = below[a + t]
                bound = (length[w] - length[x] - 1) // 2 if x != w else 0
                for d in range(D):
                    row[d] = acc[t, d]
                    acc[t, d] = 0
                    if d > bound and row[d] != 0:
                        err[0] = 2
                if x == w:
                    row[0] = 1
                g = _lookup(row, polys, table, mask)
                if g >= 0:
                    pid[a + t] = np.uint32(g)
                else:
                    # chunk-local new polynomial
                    slot = np.int64(_hash(row) & np.uint64(lmask))
                    found = -1
                    while ltable[slot] != -1:
                        li = ltable[slot]
                        same = True
                        for d in range(D):
                            if new_rows[c, li, d] != row[d]:
                                same = False
                                break
                        if same:
                            found = li
                            break
                        slot = (slot + 1) & lmask
                    if found < 0:
                        found = new_count[c]
                        if found >= cap:
                            err[0] = 1
                            found = 0
                        else:
                            for d in range(D):
                                new_rows[c, found, d] = row[d]
                            new_count[c] = found + 1
                            ltable[slot] = found
                    pid[a + t] = FLAG | np.uint32(c << 24) | np.uint32(found)
                pos[x] = -1


@njit(parallel=True, cache=True)
def _patch_level(ws, start, size, pid, remap, cap):
    for k in prange(ws.shape[0]):
        w = ws[k]
        a, m = start[w], size[w]
        for t in range(m):
            p = pid[a + t]
            if p & FLAG:
                c = (p >> np.uint32(24)) & np.uint32(127)
                li = p & np.uint32(0xFFFFFF)
                pid[a + t] = np.uint32(remap[c * cap + li])


# ---------------------------------------------------------------------------
# Coefficients and identity check
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _c_coeffs(start, size, below, pid, val1, ctype, G):
    L = G.shape[0]
    N = start.shape[0]
    out = np.zeros((L, N), dtype=np.int64)
    for w in prange(N):
        a, m = start[w], size[w]
        for t in range(m):
            v = val1[pid[a + t]]
            cl = ctype[below[a + t]]
            for l in range(L):
                out[l, w] += v * G[l, cl]
    return out


@njit(parallel=True, cache=True)
def _identity(start, size, below, pid, val1, length, w0t, C, nchunks):
    L, N = C.shape
    priv = np.zeros((nchunks, L, N), dtype=np.int64)
    per = (N + nchunks - 1) // nchunks
    for c in prange(nchunks):
        for y in range(c * per, min(N, (c + 1) * per)):
            w = w0t[y]
            a, m = start[y], size[y]
            for t in range(m):
                x = below[a + t]
                v = w0t[x]
                val = val1[pid[a + t]]
                if (length[y] - length[x]) % 2 == 1:
                    val = -val
                for l in range(L):
                    priv[c, l, v] += val * C[l, w]
    return priv.sum(axis=0)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class BigKL:
    """All KL polynomials of S_n, built level by level in parallel.

    Attributes: ``tables`` (BigTables), ``start``/``size`` (int64), ``below`` (int32),
    ``pid`` (uint32), ``polys`` (int32 rows), ``npairs``, ``timings``.
    """

    def __init__(self, n: int, cap: int = 1 << 12, max_polys: int = 1 << 22,
                 table_bits: int = 24, verbose: bool = False):
        t0 = time.time()
        self.n = n
        T = self.tables = BigTables(n)
        self.timings = {"tables": time.time() - t0}
        t = time.time()
        self.size = interval_sizes(T)
        self.start = np.concatenate([[0], np.cumsum(self.size)[:-1]]).astype(np.int64)
        self.npairs = int(self.size.sum())
        self.timings["sizes"] = time.time() - t
        if verbose:
            print(f"  S_{n}: {T.N} elements, {self.npairs} Bruhat pairs; allocating "
                  f"{self.npairs * 8 / 1e9:.1f} GB", flush=True)
        self.below = np.empty(self.npairs, dtype=np.int32)
        self.pid = np.zeros(self.npairs, dtype=np.uint32)
        nchunks = max(1, get_num_threads() * 4)
        n1 = max(n - 1, 1)
        t = time.time()
        e = int(T.order[0])
        self.below[self.start[e]] = e
        for l in range(1, len(T.level_start) - 1):
            ws = T.order[T.level_start[l]:T.level_start[l + 1]]
            _fill_level(ws, T.left, T.length, n1, self.start, self.size, self.below, T.N,
                        min(nchunks, len(ws)))
        self.timings["intervals"] = time.time() - t

        D = (n * (n - 1) // 2) // 2 + 2
        polys = np.zeros((max_polys, D), dtype=np.int32)
        npolys = np.zeros(1, dtype=np.int64)
        table = np.full(1 << table_bits, -1, dtype=np.int64)
        mask = (1 << table_bits) - 1
        one = np.zeros(D, dtype=np.int64)
        one[0] = 1
        _insert(one, polys, npolys, table, mask)
        self.pid[self.start[e]] = 0
        t = time.time()
        for l in range(1, len(T.level_start) - 1):
            ws = T.order[T.level_start[l]:T.level_start[l + 1]]
            ch = min(nchunks, len(ws), 127)
            while True:                       # retry the level with a larger buffer if needed
                new_rows = np.zeros((ch, cap, D), dtype=np.int64)
                new_count = np.zeros(ch, dtype=np.int64)
                err = np.zeros(1, dtype=np.int64)
                _kl_level(ws, T.left, T.length, n1, self.start, self.size, self.below,
                          self.pid, polys, table, mask, D, ch, cap, new_rows, new_count, err)
                if err[0] != 1:
                    break
                cap *= 2
                if verbose:
                    print(f"  level {l}: chunk buffer full, retrying with cap {cap}", flush=True)
            if err[0] == 2:
                raise ArithmeticError("degree bound violated: the recursion is wrong")
            remap = np.zeros(ch * cap, dtype=np.int64)
            for c in range(ch):
                for li in range(int(new_count[c])):
                    g = _insert(new_rows[c, li], polys, npolys, table, mask)
                    if g < 0:
                        raise MemoryError("global polynomial table full; raise max_polys")
                    remap[c * cap + li] = g
            _patch_level(ws, self.start, self.size, self.pid, remap, cap)
            if verbose and l % 4 == 0:
                print(f"  level {l}: {int(npolys[0])} distinct polynomials, "
                      f"{time.time() - t:.0f} s", flush=True)
        self.timings["kl"] = time.time() - t
        self.polys = polys[:int(npolys[0])].copy()
        self.values_at_one = self.polys.sum(axis=1).astype(np.int64)

    def sanity(self) -> Dict[str, bool]:
        return {"constant_term_one": bool(np.all(self.polys[:, 0] == 1)),
                "nonnegative_coefficients": bool(np.all(self.polys >= 0))}

    def c_coefficients_scaled(self):
        """(C, scale): C[l, w] = scale * c_{lambda_l, w}, after the exact identity check."""
        T = self.tables
        parts, g = monomial_class_functions(self.n)
        assert tuple(parts) == tuple(T.parts)
        scale = math.lcm(*[x.denominator for row in g for x in row])
        G = np.array([[int(x * scale) for x in row] for row in g], dtype=np.int64)
        t = time.time()
        C = _c_coeffs(self.start, self.size, self.below, self.pid, self.values_at_one,
                      T.ctype, G)
        bound = int(np.abs(C).max()) * int(self.values_at_one.max()) * int(self.size.max())
        if bound >= 2 ** 62:
            raise OverflowError("identity check could overflow int64")
        Tm = _identity(self.start, self.size, self.below, self.pid, self.values_at_one,
                       T.length, T.w0_times, C, max(1, get_num_threads()))
        if not np.array_equal(Tm, G[:, T.ctype]):
            raise ArithmeticError("KL expansion identity fails")
        self.timings["coefficients+identity"] = time.time() - t
        return C, scale
