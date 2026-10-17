#!/usr/bin/env python3
"""Vertex-1 (rooted) refinement of Haiman's q = 1 conjecture: Conjecture 10.2 of the hooks paper, and its match with
Abreu-Nigro's g-functions.

Usage
-----
    .venv/bin/python notes/rooted_vertex.py --n 7              # Conjecture 10.2 at every u in S_7 (~10 s; n = 8 ~2 min)
    .venv/bin/python notes/rooted_vertex.py --n 6 --an         # + exact comparison with Abreu-Nigro g_{m,k}(x;1)
    .venv/bin/python notes/rooted_vertex.py --n 6 --relations  # + the doubling relations determine G_1 from codominant u

Mathematics (Section 10 of the hooks paper, paper/hooks_kl.tex)
----------------------------------------------------------------
Brick tabloids: phi^lambda(sigma) = (-1)^{n-len(lambda)} sgn(sigma) N_lambda(sigma), N_lambda = number of cuttings of the
cycles of sigma into arcs (runs v, sigma v, ...) of lengths lambda.  For a vertex v, a >= 1 and mu |- n - a put
    G^{(a;mu)}_v(sigma) = (-1)^{n-1-len(mu)} sgn(sigma) #{ordered placements: an arc of length a starting at v, then
                           arcs of the non-singleton parts of mu in a fixed order, all disjoint}.
The arc containing v gives the exact decomposition (any fixed v; K = a G pointwise, by the automorphism sigma^j)
    phi^lambda = sum over distinct parts a of lambda of  a * G^{(a; lambda - a)}_v / aut(lambda - a),
aut = product of factorials of the multiplicities of the non-singleton parts.

Conjecture 10.2.  For v in {1, n} (the extreme vertices) and all a, mu: G^{(a;mu)}_v(C'_u(1)) >= 0 for every u.
It implies Haiman's conjecture at q = 1.  Checked for every u in S_n, n <= 8.  At interior vertices it fails
(roughly half of the (a; mu) fail at each interior v).

At codominant u (Hessenberg function m(i) = max u(1..i)):
    a * G^{(a;mu)}_1(C'_u(1)) / aut(mu) = a * [e_mu] g_{m, n-a}(x; 1),
where g_{m,k} are Abreu-Nigro's g-functions (decomposition theorem for Hess_m -> P^{n-1}, (V_i) -> V_1), in the form
(2.2) of Huh-Hwang-Kim-Kim-Oh (arXiv:2504.09123).  Checked exactly for n <= 6 (--an).  HHKKO prove g_{m,k}(x;1)
e-positive, so Conjecture 10.2 holds at every codominant u.  For general u it is the extension of Abreu-Nigro's
g-positivity to all Kazhdan-Lusztig characters (Lusztig varieties); compare Abreu-Nigro, arXiv:2212.13497, Conj. 1.4.

Doubling relations (--relations).  Let s = s_l with l >= 2 (1-indexed), so that s fixes vertex 1.  If sw < w < ws, then
C'_w C'_s = C'_w + C'_w s and C'_w s = s C'_w s (since s C'_w = C'_w), and conjugation by s fixes vertex 1, so
    G_1(C'_{ws}) + sum_{z < w, zs < z} mu(z,w) G_1(C'_z) = 2 G_1(C'_w),
and symmetrically on the left when ws < w < sw (with sz < z).  The check: (1) every G^{(a;mu)}_1 satisfies these
relations exactly; (2) the space of all functions on S_n satisfying them has dimension the number of pairs (a;mu), so
the G^{(a;mu)}_1 span it; (3) a solution vanishing at every codominant u is zero, so the codominant values determine
every value.  Ranks are computed modulo a prime p.  Since rank mod p <= rank over Q, a nullity mod p equal to the
number of linearly independent G's (independence also checked mod p) gives (2) exactly, and full rank mod p in (3)
gives full rank over Q.
"""
import argparse, itertools, os, sys
from fractions import Fraction
from collections import Counter
from math import factorial
import numpy as np

from math import factorial
from collections import Counter
from numba import njit
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from monimm.kl_fast import FastKL, evaluate_on_kl_basis
from monimm.combinatorics import partitions, monomial_class_functions


@njit
def place(p, used, parts, j):
    """Number of ordered ways to place arcs parts[j:], each starting at an unused vertex, disjoint."""
    if j == parts.shape[0]:
        return 1
    n = p.shape[0]
    L = parts[j]
    tot = 0
    for v in range(n):
        if used[v]:
            continue
        # walk L steps; fail if we hit a used vertex or return to v before finishing (arc longer than cycle)
        ok = True
        x = v
        for s in range(L):
            if used[x] or (s > 0 and x == v):
                ok = False
                break
            x = p[x]
        if not ok:
            continue
        x = v
        for s in range(L):
            used[x] = True
            x = p[x]
        tot += place(p, used, parts, j + 1)
        x = v
        for s in range(L):
            used[x] = False
            x = p[x]
    return tot


@njit
def all_G(perms, parts, sg, eps):
    N, n = perms.shape
    G = np.zeros((n, N), np.int64)
    used = np.zeros(n, np.bool_)
    for r in range(N):
        p = perms[r]
        for v in range(n):
            L = parts[0]
            ok = True
            x = v
            for s in range(L):
                if used[x] or (s > 0 and x == v):
                    ok = False
                    break
                x = p[x]
            if not ok:
                continue
            x = v
            for s in range(L):
                used[x] = True
                x = p[x]
            G[v, r] = eps * sg[r] * place(p, used, parts, 1)
            x = v
            for s in range(L):
                used[x] = False
                x = p[x]
    return G



@njit
def all_K(perms, parts, sg, eps):
    N, n = perms.shape
    K = np.zeros((n, N), np.int64)
    used = np.zeros(n, np.bool_)
    for r in range(N):
        p = perms[r]
        inv = np.empty(n, np.int64)
        for i in range(n): inv[p[i]] = i
        for v in range(n):
            tot = 0
            st = v
            for j in range(parts[0]):              # the arc containing v starts at st = sigma^{-j}(v)
                if j > 0:
                    st = inv[st]
                    if st == v: break              # cycle shorter than j+1
                L = parts[0]; ok = True; x = st
                for s in range(L):
                    if used[x] or (s > 0 and x == st): ok = False; break
                    x = p[x]
                if not ok: continue
                x = st
                for s in range(L): used[x] = True; x = p[x]
                tot += place(p, used, parts, 1)
                x = st
                for s in range(L): used[x] = False; x = p[x]
            K[v, r] = eps * sg[r] * tot
    return K



def sgnp(p):
    s = 1; seen = [False] * len(p)
    for i in range(len(p)):
        if not seen[i]:
            j = i; L = 0
            while not seen[j]: seen[j] = True; j = p[j]; L += 1
            s *= (-1) ** (L - 1)
    return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, required=True); ap.add_argument("--an", action="store_true")
    ap.add_argument("--relations", action="store_true")
    args = ap.parse_args(); n = args.n
    f = FastKL(n); Tb = f.tables
    perms = np.array(Tb.perms, dtype=np.int64)
    sg = np.array([sgnp(p) for p in Tb.perms], dtype=np.int64)
    posv = Counter(); tot = 0; bad = []
    for a in range(1, n + 1):
        for mu in (partitions(n - a) if n > a else [()]):
            mu = tuple(mu)
            G = all_G(perms, np.array([a] + [x for x in mu if x >= 2], dtype=np.int64), sg, (-1) ** (n - 1 - len(mu)))
            E = evaluate_on_kl_basis(f, G)
            ok = (E >= 0).all(axis=1); tot += 1
            for v in range(n): posv[v] += int(ok[v])
            if not (ok[0] and ok[n - 1]): bad.append((a, mu))
    print(f"n={n}: {tot} pairs (a;mu); KL-positive pairs per vertex v=1..n: {[posv[v] for v in range(n)]}")
    print(f"  Conjecture 10.2 (extreme vertices) holds for all u: {not bad}", bad[:5])
    if args.an:
        an_check(n, f, Tb, perms, sg)
    if args.relations:
        relations_check(n, f, Tb, perms, sg)


def relations_check(n, f, Tb, perms, sg, p=2147483629):
    """The doubling relations for vertex 1 (see the module docstring): checks (1), (2), (3)."""
    import flint
    N, length, rank = Tb.N, Tb.length, Tb.rank

    def mus(wr):
        """[(z, mu(z,w))] for z < w with mu(z,w) != 0 (ranks)."""
        out = []
        a0, m = f.start[wr], f.size[wr]
        for k in range(m):
            z = int(f.below[a0 + k])
            d = int(length[wr] - length[z])
            if z == wr or d % 2 == 0:
                continue
            row = f.polys[f.pid[a0 + k]]
            deg = (d - 1) // 2
            c = int(row[deg]) if deg < len(row) else 0
            if c:
                out.append((z, c))
        return out

    rows = []
    for wr in range(N):
        w = Tb.perms[wr]
        pos = {x: i for i, x in enumerate(w)}
        M = None
        for l in range(1, n - 1):                    # s = s_{l+1} (1-indexed) swaps l, l+1; fixes vertex 1
            right_up = w[l] < w[l + 1]                # ws > w
            left_up = pos[l] < pos[l + 1]             # sw > w
            if right_up == left_up:
                continue                              # need sw < w < ws, or ws < w < sw
            M = mus(wr) if M is None else M
            row = {wr: -2}
            if right_up:                              # sw < w < ws
                ws = list(w); ws[l], ws[l + 1] = ws[l + 1], ws[l]
                row[rank[tuple(ws)]] = row.get(rank[tuple(ws)], 0) + 1
                for z, c in M:
                    zz = Tb.perms[z]
                    if zz[l] > zz[l + 1]:
                        row[z] = row.get(z, 0) + c
            else:                                     # ws < w < sw
                sw = tuple(l + 1 if x == l else l if x == l + 1 else x for x in w)
                row[rank[sw]] = row.get(rank[sw], 0) + 1
                for z, c in M:
                    zz = Tb.perms[z]
                    zpos = {x: i for i, x in enumerate(zz)}
                    if zpos[l] > zpos[l + 1]:
                        row[z] = row.get(z, 0) + c
            rows.append(row)

    # the vertex-1 functions G^{(a;mu)}_1 on the KL basis
    Gs = []
    for a in range(1, n + 1):
        for mu in (partitions(n - a) if n > a else [()]):
            mu = tuple(mu)
            G = all_G(perms, np.array([a] + [x for x in mu if x >= 2], dtype=np.int64), sg, (-1) ** (n - 1 - len(mu)))
            Gs.append(evaluate_on_kl_basis(f, G)[0])
    k = len(Gs)
    satisfied = all(sum(c * int(g[j]) for j, c in row.items()) == 0 for row in rows for g in Gs)

    def rank_mod_p(rowlist):
        if not rowlist:
            return 0
        mat = flint.nmod_mat(len(rowlist), N, [int(r.get(j, 0)) % p for r in rowlist for j in range(N)], p)
        return mat.rank()

    rank_R = rank_mod_p(rows)
    rank_G = flint.nmod_mat(k, N, [int(x) % p for g in Gs for x in g], p).rank()
    nullity_bound = N - rank_R                        # >= the nullity over Q
    spans = nullity_bound == rank_G                   # then nullity over Q = rank_G = number of independent G's
    codom = [r for r, u in enumerate(Tb.perms)
             if not any(u[i] > u[k2] > u[j] for i in range(n) for j in range(i + 1, n) for k2 in range(j + 1, n))]
    unit = [{r: 1} for r in codom]
    injective = rank_mod_p(rows + unit) == N
    print(f"n={n}: {len(rows)} relations; all {k} G^(a;mu)_1 satisfy them: {satisfied}; "
          f"solution space dimension {nullity_bound} (G's independent: {rank_G}, spanning: {spans}); "
          f"codominant values ({len(codom)} u) determine the solution: {injective}")

def an_check(n, f, Tb, perms, sg):
    from monimm.combinatorics import monomial_class_functions


    def z(nu):
        r = 1
        for k, c in Counter(nu).items(): r *= k ** c * factorial(c)
        return r
    # e-coefficients of p_rho: [e_lam] p_rho = eps_rho * phi^lam(rho), eps_rho = (-1)^{|rho| - len(rho)}
    PHI = {}
    for k in range(0, n + 1):
        if k == 0: continue
        parts_k, G_k = monomial_class_functions(k)
        PHI[k] = ({tuple(p): i for i, p in enumerate(parts_k)}, G_k, [tuple(p) for p in parts_k])
    def e_coeffs_of_p(rho):
        k = sum(rho)
        if k == 0: return {(): Fraction(1)}
        idx, Gk, plist = PHI[k]
        rr = tuple(sorted(rho, reverse=True)); eps = (-1) ** (k - len(rho))
        return {lam: eps * Gk[idx[lam]][idx[rr]] for lam in plist if Gk[idx[lam]][idx[rr]] != 0}

    def g_e_coeffs(mfun, k):
        """[e_mu] g_{m,k}, as dict mu -> Fraction."""
        out = Counter()
        for sig in itertools.permutations(range(n)):
            if any(sig[i] > mfun[i] for i in range(n)): continue
            # cycles, tau_1 containing 0
            seen = set(); cyc = []
            for i in range(n):
                if i in seen: continue
                c = [i]; seen.add(i); j = sig[i]
                while j != i: c.append(j); seen.add(j); j = sig[j]
                cyc.append(c)
            t1 = len(next(c for c in cyc if 0 in c)); rest = [len(c) for c in cyc if 0 not in c]
            j = t1 - n + k
            if j < 0: continue
            sign = (-1) ** j
            # omega(p_rest) = eps_rest p_rest ; h_j = sum_{nu |- j} p_nu / z_nu
            eps_r = (-1) ** (sum(rest) - len(rest))
            nus = [()] if j == 0 else [tuple(p) for p in PHI[j][2]]
            for nu in nus:
                coef = Fraction(sign * eps_r, z(nu) if nu else 1)
                for mu, c in e_coeffs_of_p(tuple(rest) + nu).items():
                    out[mu] += coef * c
        return {mu: c for mu, c in out.items() if c}

    def has(p, pat):
        k = len(pat)
        return any(all((p[I[a]] < p[I[b]]) == (pat[a] < pat[b]) for a in range(k) for b in range(a + 1, k))
                   for I in itertools.combinations(range(n), k))
    codom = [tuple(p) for p in Tb.perms if not has(p, (2, 0, 1))]
    aut = lambda mu: int(np.prod([factorial(c) for c in Counter([x for x in mu if x >= 2]).values()])) if mu else 1
    Kv = {}
    for a in range(1, n + 1):
        for mu in (partitions(n - a) if n > a else [()]):
            mu = tuple(mu)
            Kv[(a, mu)] = evaluate_on_kl_basis(f, all_K(perms, np.array([a] + [x for x in mu if x >= 2], dtype=np.int64), sg,
                                                        (-1) ** (n - 1 - len(mu))))[0]
    match = Counter(); ex = []
    for u in codom:
        mfun = [max(u[:i + 1]) for i in range(n)]
        r = Tb.rank[u]
        for a in range(1, n + 1):
            g = g_e_coeffs(mfun, n - a)
            for mu in (partitions(n - a) if n > a else [()]):
                mu = tuple(mu)
                ours = Fraction(int(Kv[(a, mu)][r]), aut(mu))
                theirs = a * g.get(mu, Fraction(0))
                match[ours == theirs] += 1
                if ours != theirs and len(ex) < 5: ex.append((''.join(str(x + 1) for x in u), a, mu, ours, theirs))
    print(f"n={n}: codominant {len(codom)}; K_1^(a;mu)(C'_u)/aut(mu) == a [e_mu] g_(m,n-a): {dict(match)}", ex)


if __name__ == "__main__":
    main()
