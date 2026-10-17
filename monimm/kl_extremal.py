"""Kazhdan-Lusztig polynomials on extremal pairs only, and c_{lambda,w} for S_10.

Why a new engine
----------------
:mod:`monimm.kl_big` stores P_{x,w} for every Bruhat pair x <= w.  S_9 has 1.05e10
such pairs (86 GB); S_10 has about 1e12, which is out of reach.  This module stores
far fewer pairs by using the classical invariance of KL polynomials under
descents (Kazhdan-Lusztig 1979, (2.3.g); this is how du Cloux's Coxeter3 works):

    if s is a left descent of w, then P_{x,w} = P_{sx,w} for all x;
    if t is a right descent of w, then P_{x,w} = P_{xt,w} for all x.

Write I = D_L(w), J = D_R(w).  P_{x,w} is constant on each double coset
W_I x W_J, and {x <= w} is a union of such double cosets (lifting property).
So it suffices to know P_{x,w} for the *extremal* x: the maximal elements of
the double cosets, which are exactly the x <= w with D_L(x) >= I and D_R(x) >= J.
For any y <= w write y^ = max(W_I y W_J); then P_{y,w} = P_{y^,w}.

Sizes (exact, computed by :func:`extremal_sizes`):

    n = 9:  33,850,056 extremal pairs   (10,501,351,657 Bruhat pairs)
    n = 10: 1,252,173,030 extremal pairs (about 1e12 Bruhat pairs)

Double cosets as contingency tables
-----------------------------------
W_I and W_J are Young subgroups.  Left multiplication by W_I permutes *values*
within the blocks beta of I, and right multiplication by W_J permutes *positions*
within the blocks alpha of J.  So the double coset of y is determined by its
table T(y)[a, b] = #{positions i in block alpha_a : y(i) in block beta_b}, a
non-negative integer matrix with row sums alpha and column sums beta, and every
such matrix occurs.  The maximal element of the double coset with table T puts,
in each position block, the values in decreasing order, taking from each value
block b its T[a, b] largest unused values, with earlier position blocks receiving
larger values (:func:`_max_of_table`).

Bruhat order on maximal double coset representatives is the tableau criterion
restricted to the coarse grid: x <= w iff the corner sums
S_T(a, b) = sum_{a' <= a, b' >= b} T[a', b'] satisfy S_{T(x)} <= S_{T(w)} entrywise.
So the extremal x <= w are enumerated directly, as the tables dominated by T(w)
(:func:`_tables_below`), without ever touching the Bruhat interval.  This
enumeration is checked against brute force for n <= 6 in the tests.

The recursion on extremal pairs
-------------------------------
Pick s in D_L(w) and put v = s w.  For x <= w with s x < x (true for every
extremal x, since s is in D_L(w) <= D_L(x)) the standard recursion reads

    P_{x,w} = P_{sx,v} + q P_{x,v} - sum_{z in M} mu(z,v) q^{(l(w)-l(z))/2} P_{x,z},

with M = {z < v : s z < z, mu(z,v) != 0}.  Every P on the right is looked up as
P_{y,u} = P_{y^,u} (y^ for the descents of u) in the stored extremal list of u,
which is sorted; y^ not in the list means y is not <= u, so P_{y,u} = 0.

The set M is found from the descent theorem (Kazhdan-Lusztig 1979, (2.3.e)): if
mu(z,v) != 0, z < v and l(v) - l(z) > 1, then D_L(v) <= D_L(z) and
D_R(v) <= D_R(z).  So the elements of M are either covers z of v in Bruhat order
(mu = 1) or extremal elements of v's own list (mu read off the stored polynomial).

Coefficients c_{lambda,w}
-------------------------
By the KL inversion formula (see :mod:`monimm.kazhdan_lusztig`)

    c_{lambda,w} = sum_{v <= w} P_{v,w}(1) g_lambda(v)
                 = sum_{x extremal for w} P_{x,w}(1) G_lambda(I, J, x),
    G_lambda(I, J, x) = sum_{v in W_I x W_J} g_lambda(cycle type of v).

G depends on w only through (I, J).  We group the w by (I, J) (169,438 classes at
n = 10) and, for each class, run once through all of S_n by depth-first search,
identifying the double coset of each v by an additive hash of its table that is
updated one position at a time.  The count of each double coset is then
checked against |W_I x W_J| = prod alpha_a! prod beta_b! / prod T_ab!, which
rules out hash collisions (see :func:`_c_by_class`).

Checks
------
* n <= 9: c_{lambda,w} equals the full-interval engines (:mod:`monimm.kl_fast`,
  ``results/big_n9.npz``); at small n the stored P_{x,w} equal those of
  :mod:`monimm.kazhdan_lusztig`.
* Every n: constant term 1, non-negative coefficients, the degree bound
  deg P_{x,w} <= (l(w) - l(x) - 1)/2, the J_mu values
  c_{lambda, w_0(mu)} = mu! delta_{lambda mu} (Lemma 6.1 of the hooks paper (Lemma 3.1 of the n <= 10 paper), since
  C'_{w_0(mu)}(1) is the sum over the Young subgroup S_mu), the symmetries
  c_{lambda,w} = c_{lambda,w^{-1}} = c_{lambda,w_0 w w_0}, which the left-descent
  recursion does not build in, and :func:`sampled_identity`, the coefficient of
  x^u in sum_w c_{lambda,w} Imm_w for random u, which tests the KL inversion
  formula on the stored polynomials.
"""

from __future__ import annotations

import math
import time
from typing import Dict

import numpy as np
from numba import njit, prange, get_num_threads

from .combinatorics import monomial_class_functions
from .kl_big import BigTables, FLAG, _hash, _lookup, _insert, _patch_level


# ---------------------------------------------------------------------------
# Descents, blocks, tables
# ---------------------------------------------------------------------------

@njit(cache=True)
def _descents(p):
    """(L, R) bit masks: bit i of L iff s_i p < p (values i, i+1 inverted),
    bit i of R iff p s_i < p (positions i, i+1 inverted)."""
    n = p.shape[0]
    pos = np.empty(n, np.int64)
    for i in range(n):
        pos[p[i]] = i
    L = 0
    R = 0
    for i in range(n - 1):
        if p[i] > p[i + 1]:
            R |= 1 << i
        if pos[i] > pos[i + 1]:
            L |= 1 << i
    return L, R


@njit(parallel=True, cache=True)
def _all_descents(perms):
    N = perms.shape[0]
    Ld = np.zeros(N, np.int32)
    Rd = np.zeros(N, np.int32)
    for k in prange(N):
        L, R = _descents(perms[k])
        Ld[k] = L
        Rd[k] = R
    return Ld, Rd


@njit(cache=True)
def _block_ids(mask, n, out):
    """out[i] = block of i in the composition of n cut by the mask; returns #blocks."""
    b = 0
    out[0] = 0
    for i in range(n - 1):
        if not (mask >> i) & 1:
            b += 1
        out[i + 1] = b
    return b + 1


@njit(cache=True)
def _sizes(ids, m, out):
    for k in range(m):
        out[k] = 0
    for i in range(ids.shape[0]):
        out[ids[i]] += 1


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


@njit(cache=True)
def _max_of_table(T, r, c, vb_top, out):
    """Maximal permutation with table T (rows: position blocks, cols: value blocks).

    vb_top[b] = largest value in value block b.  Position blocks are filled in
    order; each takes T[a, b] values from block b, largest unused first, and its
    values are written in decreasing order (higher value blocks first)."""
    top = vb_top.copy()
    i = 0
    for a in range(r):
        for b in range(c - 1, -1, -1):
            for _ in range(T[a, b]):
                out[i] = top[b]
                top[b] -= 1
                i += 1


@njit(cache=True)
def _canon_rank(y, L, R, fact, pb, vb, T, vb_top, buf):
    """Rank of max(W_L y W_R).  pb, vb, T, vb_top, buf are scratch arrays."""
    n = y.shape[0]
    r = _block_ids(R, n, pb)
    c = _block_ids(L, n, vb)
    for a in range(r):
        for b in range(c):
            T[a, b] = 0
    for i in range(n):
        T[pb[i], vb[y[i]]] += 1
    for j in range(n):
        vb_top[vb[j]] = j          # increasing j: ends at the largest value of the block
    _max_of_table(T, r, c, vb_top[:c], buf)
    return _rank(buf, fact)


@njit(cache=True)
def _bsearch(a, lo, hi, key):
    """Index of key in the sorted slice a[lo:hi], or -1."""
    while lo < hi:
        mid = (lo + hi) >> 1
        v = a[mid]
        if v < key:
            lo = mid + 1
        elif v > key:
            hi = mid
        else:
            return mid
    return -1


@njit(cache=True)
def _tables_below(p, fact, out, off):
    """Enumerate the extremal x <= p as tables dominated by T(p).

    Returns the number of tables.  If out has positive length, writes the rank of
    each maximal representative to out[off:off + count] (unsorted)."""
    n = p.shape[0]
    L, R = _descents(p)
    pb = np.empty(n, np.int64)
    vb = np.empty(n, np.int64)
    r = _block_ids(R, n, pb)
    c = _block_ids(L, n, vb)
    alpha = np.zeros(r, np.int64)
    beta = np.zeros(c, np.int64)
    _sizes(pb, r, alpha)
    _sizes(vb, c, beta)
    vb_top = np.zeros(c, np.int64)
    for j in range(n):
        vb_top[vb[j]] = j
    # corner sums of T(p)
    Tw = np.zeros((r, c), np.int64)
    for i in range(n):
        Tw[pb[i], vb[p[i]]] += 1
    SW = np.zeros((r, c), np.int64)
    for a in range(r):
        acc = 0
        for b in range(c - 1, -1, -1):
            acc += Tw[a, b]
            SW[a, b] = acc + (SW[a - 1, b] if a > 0 else 0)
    # depth-first search over the cells, row by row, right to left
    S = np.zeros((r, c), np.int64)
    T = np.zeros((r, c), np.int64)
    colrem = beta.copy()
    rowrem = np.zeros(r, np.int64)
    rowrem[0] = alpha[0]
    ncell = r * c
    val = np.full(ncell, -1, np.int64)
    buf = np.empty(n, np.int64)
    write = out.shape[0] > 0
    idx = 0
    count = 0
    while idx >= 0:
        a = idx // c
        b = c - 1 - (idx % c)
        prev = val[idx]
        if prev >= 0:
            rowrem[a] += prev
            colrem[b] += prev
        if b == 0:
            # the first column takes what is left of the row; only one choice
            if prev >= 0:
                val[idx] = -1
                idx -= 1
                continue
            v = rowrem[a]
            if v > colrem[b]:
                val[idx] = -1
                idx -= 1
                continue
        else:
            v = prev + 1
            if v > rowrem[a] or v > colrem[b]:
                val[idx] = -1
                idx -= 1
                continue
        rs = v
        if b + 1 < c:
            rs += S[a, b + 1] - (S[a - 1, b + 1] if a > 0 else 0)
        s = rs + (S[a - 1, b] if a > 0 else 0)
        if s > SW[a, b]:
            # larger values only increase the corner sum
            val[idx] = -1
            idx -= 1
            continue
        val[idx] = v
        T[a, b] = v
        rowrem[a] -= v
        colrem[b] -= v
        S[a, b] = s
        if idx == ncell - 1:
            if write:
                _max_of_table(T, r, c, vb_top, buf)
                out[off + count] = _rank(buf, fact)
            count += 1
            continue
        idx += 1
        if idx % c == 0:
            rowrem[idx // c] = alpha[idx // c]
    return count


@njit(parallel=True, cache=True)
def _count_all(perms, fact):
    N = perms.shape[0]
    out = np.zeros(N, np.int64)
    empty = np.empty(0, np.int32)
    for k in prange(N):
        out[k] = _tables_below(perms[k], fact, empty, 0)
    return out


@njit(parallel=True, cache=True)
def _fill_all(perms, fact, start, xs):
    N = perms.shape[0]
    for k in prange(N):
        a = start[k]
        m = _tables_below(perms[k], fact, xs, a)
        xs[a:a + m] = np.sort(xs[a:a + m])


def extremal_sizes(n: int) -> np.ndarray:
    """size[w] = number of extremal x <= w (w indexed by lexicographic rank)."""
    T = BigTables(n)
    return _count_all(T.perms, T.fact)


# ---------------------------------------------------------------------------
# The KL recursion on extremal pairs
# ---------------------------------------------------------------------------

@njit(cache=True)
def _first_left_descent(Ld, w, n1):
    for i in range(n1):
        if (Ld[w] >> i) & 1:
            return i
    return -1


@njit(cache=True)
def _lookup_pid(y, u, perms, Ld, Rd, fact, start, size, xs, pb, vb, T, vb_top, buf):
    """Index into xs/pid of P_{y,u}, or -1 if y is not <= u."""
    k = _canon_rank(perms[y], Ld[u], Rd[u], fact, pb, vb, T, vb_top, buf)
    return _bsearch(xs, start[u], start[u] + size[u], k)


@njit(parallel=True, cache=True)
def _kl_level(ws, perms, length, left, Ld, Rd, fact, start, size, xs, pid, polys, table,
              mask, D, nchunks, cap, new_rows, new_count, err, nmu):
    m = ws.shape[0]
    n = perms.shape[1]
    N = perms.shape[0]
    # chunk c takes ws[c], ws[c + nchunks], ...: neighbours in ws cost about the
    # same, so interleaving balances the threads better than contiguous blocks
    for c in prange(nchunks):
        if c >= m:
            continue
        pb = np.empty(n, np.int64)
        vb = np.empty(n, np.int64)
        T = np.zeros((n, n), np.int64)
        vb_top = np.zeros(n, np.int64)
        buf = np.empty(n, np.int64)
        zbuf = np.empty(n, np.int8)
        row = np.zeros(D, np.int64)
        lmask = 4095
        while (lmask + 1) < 4 * cap:
            lmask = lmask * 2 + 1
        ltable = np.full(lmask + 1, -1, np.int64)
        maxz = 4096
        zs = np.empty(maxz, np.int64)
        ms = np.empty(maxz, np.int64)
        for k in range(c, m, nchunks):
            w = ws[k]
            s = _first_left_descent(Ld, w, n - 1)
            v = left[s, w]
            # M = {z < v : s z < z, mu(z, v) != 0}
            nz = 0
            pv = perms[v]
            for i in range(n):           # covers v (i j) with v(i) > v(j), nothing in between
                for j in range(i + 1, n):
                    if pv[i] <= pv[j]:
                        continue
                    ok = True
                    for t in range(i + 1, j):
                        if pv[j] < pv[t] < pv[i]:
                            ok = False
                            break
                    if not ok:
                        continue
                    for t in range(n):
                        zbuf[t] = pv[t]
                    zbuf[i] = pv[j]
                    zbuf[j] = pv[i]
                    z = _rank(zbuf, fact)
                    if length[left[s, z]] < length[z]:
                        if nz < maxz:
                            zs[nz] = z
                            ms[nz] = 1
                        nz += 1
            av = start[v]
            for t in range(av, av + size[v]):   # extremal z of v with odd gap >= 3
                z = xs[t]
                diff = length[v] - length[z]
                if diff < 3 or diff % 2 == 0:
                    continue
                if length[left[s, z]] > length[z]:
                    continue
                e = (diff - 1) // 2
                mu = polys[pid[t], e] if e < D else 0
                if mu != 0:
                    if nz < maxz:
                        zs[nz] = z
                        ms[nz] = mu
                    nz += 1
            if nz > maxz:
                err[0] = 3
                continue
            if nz > nmu[c]:
                nmu[c] = nz
            aw = start[w]
            for t in range(aw, aw + size[w]):
                x = xs[t]
                for d in range(D):
                    row[d] = 0
                # P_{sx, v}
                j = _lookup_pid(left[s, x], v, perms, Ld, Rd, fact, start, size, xs,
                                pb, vb, T, vb_top, buf)
                if j >= 0:
                    p = pid[j]
                    for d in range(D):
                        row[d] += polys[p, d]
                # q P_{x, v}
                j = _lookup_pid(x, v, perms, Ld, Rd, fact, start, size, xs,
                                pb, vb, T, vb_top, buf)
                if j >= 0:
                    p = pid[j]
                    for d in range(D - 1):
                        row[d + 1] += polys[p, d]
                # - sum mu(z, v) q^{(l(w) - l(z))/2} P_{x, z}
                for jz in range(nz):
                    z = zs[jz]
                    if length[x] > length[z]:
                        continue
                    j = _lookup_pid(x, z, perms, Ld, Rd, fact, start, size, xs,
                                    pb, vb, T, vb_top, buf)
                    if j >= 0:
                        p = pid[j]
                        sh = (length[w] - length[z]) // 2
                        mu = ms[jz]
                        for d in range(D - sh):
                            row[d + sh] -= mu * polys[p, d]
                # degree bound and normalisation
                if x == w:
                    if row[0] != 1:
                        err[0] = 4
                    for d in range(1, D):
                        if row[d] != 0:
                            err[0] = 4
                else:
                    bound = (length[w] - length[x] - 1) // 2
                    for d in range(D):
                        if d > bound and row[d] != 0:
                            err[0] = 2
                    if row[0] != 1:
                        err[0] = 5
                g = _lookup(row, polys, table, mask)
                if g >= 0:
                    pid[t] = np.uint32(g)
                else:
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
                    pid[t] = FLAG | np.uint32(c << 24) | np.uint32(found)


# ---------------------------------------------------------------------------
# Coefficients c_{lambda, w}
# ---------------------------------------------------------------------------

@njit(cache=True)
def _table_hash(p, pb, vb, H):
    """Additive hash of the table of p: sum_i H[pb[i], vb[p[i]]] (mod 2^64)."""
    h = np.uint64(0)
    for i in range(p.shape[0]):
        h += H[pb[i], vb[p[i]]]
    return h


@njit(parallel=True, cache=True)
def _c_by_class(class_ws, class_start, perms, ctype, nct, Ld, Rd, start, size, xs, pid,
                val1, G, H, logfact, out, err):
    """For each (D_L, D_R) class: G-sums over the double cosets, then c[:, w].

    The double coset of v is identified by an additive hash of its table T(v),
    accumulated along a depth-first run through S_n in lexicographic order, so
    the run costs about one addition per node and the rank of v is a counter.
    Exactness: every needed double coset must receive exactly
    |W_I x W_J| = prod alpha_a! prod beta_b! / prod T_ab! hits.  All its elements
    do hit it, so a hash collision with another table would show up as extra
    hits (err[0] = 6), as would two needed tables with equal hashes (err[0] = 7)."""
    ncls = class_start.shape[0] - 1
    n = perms.shape[1]
    nl = G.shape[0]
    for g in prange(ncls):
        lo = class_start[g]
        hi = class_start[g + 1]
        w0 = class_ws[lo]
        L = Ld[w0]
        R = Rd[w0]
        pb = np.empty(n, np.int64)
        vb = np.empty(n, np.int64)
        r = _block_ids(R, n, pb)
        c = _block_ids(L, n, vb)
        # the needed maximal representatives, deduplicated, and their hashes
        tot = 0
        for k in range(lo, hi):
            tot += size[class_ws[k]]
        keys = np.empty(tot, np.int64)
        i = 0
        for k in range(lo, hi):
            w = class_ws[k]
            for t in range(start[w], start[w] + size[w]):
                keys[i] = xs[t]
                i += 1
        keys = np.unique(keys)
        nk = keys.shape[0]
        hmask = 1
        while hmask + 1 < 2 * nk:
            hmask = hmask * 2 + 1
        hkey = np.zeros(hmask + 1, np.uint64)
        hidx = np.full(hmask + 1, -1, np.int64)
        expect = np.zeros(nk, np.int64)
        alpha = np.zeros(r, np.int64)
        beta = np.zeros(c, np.int64)
        _sizes(pb, r, alpha)
        _sizes(vb, c, beta)
        base = 0.0
        for a in range(r):
            base += logfact[alpha[a]]
        for b in range(c):
            base += logfact[beta[b]]
        T = np.zeros((r, c), np.int64)
        for j in range(nk):
            p = perms[keys[j]]
            h = _table_hash(p, pb, vb, H)
            slot = np.int64(h & np.uint64(hmask))
            while hidx[slot] != -1:
                if hkey[slot] == h:
                    err[0] = 7
                slot = (slot + 1) & hmask
            hkey[slot] = h
            hidx[slot] = j
            for a in range(r):
                for b in range(c):
                    T[a, b] = 0
            for t in range(n):
                T[pb[t], vb[p[t]]] += 1
            lg = base
            for a in range(r):
                for b in range(c):
                    lg -= logfact[T[a, b]]
            expect[j] = np.int64(np.round(np.exp(lg)))
        # depth-first run through S_n, lexicographic order
        prof = np.zeros((nk, nct), np.int64)
        used = np.zeros(n, np.bool_)
        choice = np.full(n, -1, np.int64)
        part = np.zeros(n + 1, np.uint64)
        depth = 0
        rank = 0
        while depth >= 0:
            if choice[depth] >= 0:
                used[choice[depth]] = False
            v = choice[depth] + 1
            while v < n and used[v]:
                v += 1
            if v == n:
                choice[depth] = -1
                depth -= 1
                continue
            choice[depth] = v
            used[v] = True
            part[depth + 1] = part[depth] + H[pb[depth], vb[v]]
            if depth == n - 1:
                h = part[n]
                slot = np.int64(h & np.uint64(hmask))
                while hidx[slot] != -1:
                    if hkey[slot] == h:
                        prof[hidx[slot], ctype[rank]] += 1
                        break
                    slot = (slot + 1) & hmask
                rank += 1
            else:
                depth += 1
        # exactness of the double coset sums
        for j in range(nk):
            cnt = 0
            for ct in range(nct):
                cnt += prof[j, ct]
            if cnt != expect[j]:
                err[0] = 6
        acc = np.zeros((nk, nl), np.int64)
        for j in range(nk):
            for ct in range(nct):
                pc = prof[j, ct]
                if pc != 0:
                    for l in range(nl):
                        acc[j, l] += pc * G[l, ct]
        for k in range(lo, hi):
            w = class_ws[k]
            for t in range(start[w], start[w] + size[w]):
                j = _bsearch(keys, 0, nk, xs[t])
                v1 = val1[pid[t]]
                for l in range(nl):
                    out[l, w] += v1 * acc[j, l]


@njit(parallel=True, cache=True)
def _sampled_identity(us, perms, length, w0t, Ld, Rd, fact, start, size, xs, pid, val1, C,
                      ctype, G):
    """For each sampled u, check sum_{w <= u} (-1)^{l(u)-l(w)} P_{w0 u, w0 w}(1) c_w = g(u).

    (Coefficient of x^u in sum_w c_w Imm_w, Rhoades-Skandera's definition of Imm_w.)
    With x = w0 u and y = w0 w the sum runs over all y >= x; we scan all y."""
    nl = C.shape[0]
    N = perms.shape[0]
    n = perms.shape[1]
    bad = np.zeros(us.shape[0], np.int64)
    for k in prange(us.shape[0]):
        u = us[k]
        x = w0t[u]
        pb = np.empty(n, np.int64)
        vb = np.empty(n, np.int64)
        T = np.zeros((n, n), np.int64)
        vb_top = np.zeros(n, np.int64)
        buf = np.empty(n, np.int64)
        tot = np.zeros(nl, np.int64)
        for y in range(N):
            if length[y] < length[x]:
                continue
            j = _lookup_pid(x, y, perms, Ld, Rd, fact, start, size, xs, pb, vb, T, vb_top, buf)
            if j < 0:
                continue
            val = val1[pid[j]]
            if (length[y] - length[x]) % 2 == 1:
                val = -val
            w = w0t[y]
            for l in range(nl):
                tot[l] += val * C[l, w]
        for l in range(nl):
            if tot[l] != G[l, ctype[u]]:
                bad[k] = 1
    return bad


# A prime below 2^31, so that products of residues fit in int64.
_EXACT_PRIME = 2147483629


@njit(parallel=True, cache=True)
def _sampled_identity_exact(us, perms, length, w0t, Ld, Rd, fact, start, size, xs, pid, val1, C,
                            ctype, G):
    """As :func:`_sampled_identity`, but certified over Z rather than modulo 2^64.

    For each sampled u and each lambda it returns three things: whether the int64 sum
    (which wraps modulo 2^64) agrees with g_lambda(u), whether the sum taken modulo the
    prime p = _EXACT_PRIME agrees with g_lambda(u) mod p, and the float64 value of
    B = sum_y |P_{x,y}(1)| |c_{lambda, w0 y}|.  B bounds the absolute value of every
    partial sum.  Once B < 2^62 (well inside int64, even allowing for the relative rounding
    error of B, about 10^-9 here), no partial sum wraps, so the int64 sum is the integer
    sum and the first flag is an identity over Z.  The mod-p sum is a second, independent
    accumulation.  Agreement both mod 2^64 and mod p gives agreement modulo 2^64 p, about
    2^95, so B < 2^94 would already suffice."""
    nl = C.shape[0]
    n = perms.shape[1]
    p = _EXACT_PRIME
    bad64 = np.zeros((us.shape[0], nl), np.int64)
    badp = np.zeros((us.shape[0], nl), np.int64)
    bound = np.zeros((us.shape[0], nl), np.float64)
    N = perms.shape[0]
    for k in prange(us.shape[0]):
        u = us[k]
        x = w0t[u]
        pb = np.empty(n, np.int64)
        vb = np.empty(n, np.int64)
        T = np.zeros((n, n), np.int64)
        vb_top = np.zeros(n, np.int64)
        buf = np.empty(n, np.int64)
        tot = np.zeros(nl, np.int64)
        totp = np.zeros(nl, np.int64)
        bnd = np.zeros(nl, np.float64)
        for y in range(N):
            if length[y] < length[x]:
                continue
            j = _lookup_pid(x, y, perms, Ld, Rd, fact, start, size, xs, pb, vb, T, vb_top, buf)
            if j < 0:
                continue
            val = val1[pid[j]]
            if (length[y] - length[x]) % 2 == 1:
                val = -val
            w = w0t[y]
            vp = val % p
            for l in range(nl):
                c = C[l, w]
                tot[l] += val * c
                totp[l] = (totp[l] + vp * (c % p)) % p
                bnd[l] += abs(float(val)) * abs(float(c))
        for l in range(nl):
            g = G[l, ctype[u]]
            bad64[k, l] = tot[l] != g
            badp[k, l] = totp[l] != g % p
            bound[k, l] = bnd[l]
    return bad64, badp, bound


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class ExtremalKL:
    """KL polynomials P_{x,w} of S_n for the extremal pairs, built level by level.

    Attributes: ``tables`` (BigTables), ``Ld``/``Rd`` (descent masks), ``start``/``size``
    (int64), ``xs`` (int32, sorted ranks of the extremal x of each w), ``pid`` (uint32),
    ``polys`` (int32 rows), ``npairs``, ``timings``.
    """

    def __init__(self, n: int, cap: int = 1 << 12, max_polys: int = 1 << 22,
                 table_bits: int = 24, verbose: bool = False):
        t0 = time.time()
        self.n = n
        T = self.tables = BigTables(n)
        self.Ld, self.Rd = _all_descents(T.perms)
        self.timings = {"tables": time.time() - t0}
        t = time.time()
        self.size = _count_all(T.perms, T.fact)
        self.start = np.concatenate([[0], np.cumsum(self.size)[:-1]]).astype(np.int64)
        self.npairs = int(self.size.sum())
        if verbose:
            print(f"  S_{n}: {T.N} elements, {self.npairs} extremal pairs; allocating "
                  f"{self.npairs * 8 / 1e9:.1f} GB", flush=True)
        self.xs = np.empty(self.npairs, dtype=np.int32)
        _fill_all(T.perms, T.fact, self.start, self.xs)
        self.timings["extremal lists"] = time.time() - t
        self.pid = np.zeros(self.npairs, dtype=np.uint32)

        D = (n * (n - 1) // 2) // 2 + 2
        self.D = D
        polys = np.zeros((max_polys, D), dtype=np.int32)
        npolys = np.zeros(1, dtype=np.int64)
        table = np.full(1 << table_bits, -1, dtype=np.int64)
        mask = (1 << table_bits) - 1
        one = np.zeros(D, dtype=np.int64)
        one[0] = 1
        _insert(one, polys, npolys, table, mask)
        e = int(T.order[0])
        self.pid[self.start[e]] = 0
        nchunks = max(1, get_num_threads() * 4)
        self.max_mu = 0
        t = time.time()
        for l in range(1, len(T.level_start) - 1):
            ws = T.order[T.level_start[l]:T.level_start[l + 1]]
            ch = min(nchunks, len(ws), 127)
            while True:                       # retry the level with a larger buffer if needed
                new_rows = np.zeros((ch, cap, D), dtype=np.int64)
                new_count = np.zeros(ch, dtype=np.int64)
                err = np.zeros(1, dtype=np.int64)
                nmu = np.zeros(ch, dtype=np.int64)
                _kl_level(ws, T.perms, T.length, T.left, self.Ld, self.Rd, T.fact,
                          self.start, self.size, self.xs, self.pid, polys, table, mask,
                          D, ch, cap, new_rows, new_count, err, nmu)
                if err[0] != 1:
                    break
                cap *= 2
                if verbose:
                    print(f"  level {l}: chunk buffer full, retrying with cap {cap}", flush=True)
            if err[0] == 2:
                raise ArithmeticError("degree bound violated: the recursion is wrong")
            if err[0] == 3:
                raise MemoryError("mu-list buffer too small")
            if err[0] in (4, 5):
                raise ArithmeticError("P_{w,w} != 1 or P_{x,w}(0) != 1")
            self.max_mu = max(self.max_mu, int(nmu.max()))
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

    # -- queries -------------------------------------------------------------

    def P(self, x, w):
        """Coefficient list of P_{x,w} (x, w permutation tuples); [] if x is not <= w."""
        T = self.tables
        xr, wr = T.rank(x), T.rank(w)
        n = self.n
        scratch = (np.empty(n, np.int64), np.empty(n, np.int64), np.zeros((n, n), np.int64),
                   np.zeros(n, np.int64), np.empty(n, np.int64))
        j = _lookup_pid(xr, wr, T.perms, self.Ld, self.Rd, T.fact, self.start, self.size,
                        self.xs, *scratch)
        if j < 0:
            return []
        row = [int(c) for c in self.polys[self.pid[j]]]
        while row and row[-1] == 0:
            row.pop()
        return row

    def sanity(self) -> Dict[str, bool]:
        return {"constant_term_one": bool(np.all(self.polys[:, 0] == 1)),
                "nonnegative_coefficients": bool(np.all(self.polys >= 0))}

    def c_coefficients_scaled(self):
        """(C, scale): C[l, w] = scale * c_{lambda_l, w}."""
        T = self.tables
        parts, g = monomial_class_functions(self.n)
        assert tuple(parts) == tuple(T.parts)
        scale = math.lcm(*[x.denominator for row in g for x in row])
        G = np.array([[int(x * scale) for x in row] for row in g], dtype=np.int64)
        # The double cosets of the extremal x of w are disjoint, so
        # |c| <= max P(1) * max|G| * sum_x |W_I x W_J| <= max P(1) * max|G| * n!.
        bound = T.N * int(np.abs(G).max()) * int(self.values_at_one.max())
        if bound >= 2 ** 62:
            raise OverflowError("c coefficients could overflow int64")
        t = time.time()
        key = self.Ld.astype(np.int64) << 32 | self.Rd.astype(np.int64)
        order = np.argsort(key, kind="stable")
        sk = key[order]
        cuts = np.flatnonzero(np.diff(sk)) + 1
        class_start = np.concatenate([[0], cuts, [len(sk)]]).astype(np.int64)
        C = np.zeros((G.shape[0], T.N), dtype=np.int64)
        n = self.n
        H = np.random.default_rng(12345).integers(0, 2 ** 63, size=(n, n), dtype=np.uint64)
        H = H * np.uint64(2) + np.uint64(1)   # odd, full 64-bit range
        logfact = np.array([math.lgamma(k + 1) for k in range(n + 1)])
        err = np.zeros(1, dtype=np.int64)
        _c_by_class(order.astype(np.int64), class_start, T.perms, T.ctype.astype(np.int64),
                    len(T.parts), self.Ld, self.Rd, self.start, self.size, self.xs, self.pid,
                    self.values_at_one, G, H, logfact, C, err)
        if err[0] == 6:
            raise ArithmeticError("double coset hit count differs from its size")
        if err[0] == 7:
            raise ArithmeticError("two needed double cosets share a hash; change the seed")
        self.timings["coefficients"] = time.time() - t
        self.G = G
        return C, scale

    def sampled_identity(self, C, samples: int = 1000, seed: int = 0) -> bool:
        """KL inversion check on ``samples`` random monomials x^u (all of them if fewer).

        The sums are accumulated in int64, which wraps around, so for large n this
        is an exact check modulo 2^64 rather than over Z."""
        T = self.tables
        rng = np.random.default_rng(seed)
        us = (np.arange(T.N) if samples >= T.N
              else rng.choice(T.N, size=samples, replace=False)).astype(np.int64)
        bad = _sampled_identity(us, T.perms, T.length, T.w0_times, self.Ld, self.Rd, T.fact,
                                self.start, self.size, self.xs, self.pid, self.values_at_one,
                                C, T.ctype, self.G)
        return not bad.any()

    def g_table(self):
        """G[l, class] = scale * g_lambda_l on each cycle type, as in :meth:`c_coefficients_scaled`.

        Lets a stored coefficient table be checked without recomputing it."""
        parts, g = monomial_class_functions(self.n)
        assert tuple(parts) == tuple(self.tables.parts)
        scale = math.lcm(*[x.denominator for row in g for x in row])
        self.G = np.array([[int(x * scale) for x in row] for row in g], dtype=np.int64)
        return self.G, scale

    def sampled_identity_exact(self, C, samples: int = 1000, seed: int = 0) -> Dict[str, object]:
        """The check of :meth:`sampled_identity` (same seed, so the same u), certified over Z.

        See :func:`_sampled_identity_exact`.  ``exact`` is True when every int64 sum is
        provably free of wraparound (max bound < 2^62) and agrees with g_lambda(u); then the
        identity holds over Z for every sampled u and every lambda."""
        T = self.tables
        rng = np.random.default_rng(seed)
        us = (np.arange(T.N) if samples >= T.N
              else rng.choice(T.N, size=samples, replace=False)).astype(np.int64)
        bad64, badp, bound = _sampled_identity_exact(
            us, T.perms, T.length, T.w0_times, self.Ld, self.Rd, T.fact, self.start, self.size,
            self.xs, self.pid, self.values_at_one, C, T.ctype, self.G)
        bmax = float(bound.max())
        no_wrap = bmax < 2.0 ** 62
        return {"samples": int(us.size), "lambdas": int(C.shape[0]),
                "mismatches_mod_2^64": int(bad64.sum()),
                "mismatches_mod_p": int(badp.sum()), "p": _EXACT_PRIME,
                "max_abs_sum_bound": bmax, "log2_max_bound": float(np.log2(max(bmax, 1.0))),
                "no_wraparound": bool(no_wrap),
                "exact": bool(no_wrap and not bad64.any() and not badp.any())}
