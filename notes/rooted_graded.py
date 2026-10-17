"""Graded test of Conjecture 10.2 of the hooks paper (see notes/rooted_vertex.py).

Usage:  .venv/bin/python notes/rooted_graded.py N [pal]      (N <= 6; N = 6 takes a few minutes)

1. Graded KL characters ch(q^{l(w)/2} C'_w) = sum_{v <= w} P_{v,w}(q) ch(T_v), computed independently through
   Geck-Pfeiffer class polynomials (T_v reduced modulo [H,H] to minimal-length elements) and
   ch(T_{c_rho}) = prod_k sum_j (-1)^j q^{k-1-j} s_{(k-j,1^j)}.  Checked against c_{lambda,w} at q = 1.
2. Lifting: c_{i,mu}(q) in N[q] with c_{i,mu}(1) = G^{(n-i;mu)}_1(C'_w)/aut(mu) and
   sum_i [n-i]_q h_{n-i} sum_mu c_{i,mu}(q) h_mu = ch(q^{l/2} C'_w); with 'pal', each c_{i,mu} palindromic about
   (l(w) - (n-i-1))/2, as the decomposition theorem predicts for Abreu-Nigro's pieces ch(rho_{i,w}).
Results (2026-10-14): lift exists for every w, n <= 6 (720/720 at n = 6, 354 singular), unique for 409/720.
"""
# Graded test of Conjecture 10.2, a combinatorial form of Abreu-Nigro, arXiv:2212.13497, Conj. 1.4 (J = {2..n-1}).
# (1) graded KL characters ch(q^{l(w)/2} C'_w) = sum_{v<=w} P_{v,w}(q) ch(T_v) via Geck-Pfeiffer class polynomials,
#     with ch(T_{c_rho}) = prod_k sum_j (-1)^j q^{k-1-j} s_{(k-j,1^j)}; checked against c_{lambda,w} at q = 1.
# (2) lifting: find c_{i,mu}(q) in N[q] with c_{i,mu}(1) = G^{(n-i;mu)}_1(C'_w)/aut(mu) and
#     sum_i [n-i]_q h_{n-i} sum_mu c_{i,mu}(q) h_mu = ch(q^{l/2} C'_w); optionally palindromic about (l(w)-(n-i-1))/2.
import sys, itertools, numpy as np
from fractions import Fraction
from collections import Counter, defaultdict
from math import factorial
from functools import lru_cache
from scipy.optimize import milp, LinearConstraint, Bounds
sys.path.insert(0, '.')
exec(open('notes/rooted_vertex.py').read().split('\ndef main():')[0])
from monimm.kazhdan_lusztig import KazhdanLusztig, length
from monimm.combinatorics import monomial_class_functions, cycle_type, z_value
from monimm.ordering import character

n = int(sys.argv[1]); PAL = len(sys.argv) > 2 and sys.argv[2] == 'pal'
perms = list(itertools.permutations(range(n)))
kl = KazhdanLusztig(n)

# --- polynomials in q: dict exp -> Fraction
def padd(a, b, s=1):
    out = dict(a)
    for k, v in b.items(): out[k] = out.get(k, 0) + s * v
    return {k: v for k, v in out.items() if v}
def pmul(a, b):
    out = defaultdict(Fraction)
    for i, x in a.items():
        for j, y in b.items(): out[i + j] += x * y
    return {k: v for k, v in out.items() if v}
Q = {1: Fraction(1)}; ONE = {0: Fraction(1)}

# --- class polynomials f_v : dict rho -> poly
def simple(i):
    s = list(range(n)); s[i], s[i + 1] = s[i + 1], s[i]; return tuple(s)
S = [simple(i) for i in range(n - 1)]
def comp(a, b): return tuple(a[b[x]] for x in range(n))      # (a o b)(x) = a(b(x))
L = {p: length(p) for p in perms}
def ctype(p): return tuple(cycle_type(p))
minlen = {}
for p in perms:
    r = ctype(p); minlen[r] = min(minlen.get(r, 99), L[p])
@lru_cache(None)
def classpoly(v):
    if L[v] == minlen[ctype(v)]: return ((ctype(v), ((0, 1),)),)
    seen = {v}; frontier = [v]
    while frontier:
        nxt = []
        for u in frontier:
            for s in S:
                us = comp(comp(s, u), s)
                if L[us] == L[u] - 2:
                    a = classpoly(comp(s, u)); b = classpoly(us)
                    out = defaultdict(dict)
                    for rho, poly in a:            # (q-1) f_{su}
                        out[rho] = padd(out[rho], pmul({k: Fraction(c) for k, c in poly}, {1: Fraction(1), 0: Fraction(-1)}))
                    for rho, poly in b:            # q f_{sus}
                        out[rho] = padd(out[rho], pmul({k: Fraction(c) for k, c in poly}, Q))
                    return tuple((rho, tuple(sorted(pl.items()))) for rho, pl in out.items() if pl)
                if L[us] == L[u] and us not in seen:
                    seen.add(us); nxt.append(us)
                    if L[us] == minlen[ctype(us)]: return ((ctype(us), ((0, 1),)),)
        frontier = nxt
    raise RuntimeError(v)

# --- symmetric functions in the p basis: dict rho -> poly
parts_n, Gt = monomial_class_functions(n); pidx = {tuple(p): i for i, p in enumerate(parts_n)}
def schur_p(shape):
    k = sum(shape); return {tuple(r): {0: Fraction(character(shape, tuple(r)), z_value(tuple(r)))} for r in map(tuple, partitions(k))}
def sf_mul(A, B):
    out = defaultdict(dict)
    for ra, pa in A.items():
        for rb, pb in B.items():
            rr = tuple(sorted(ra + rb, reverse=True)); out[rr] = padd(out[rr], pmul(pa, pb))
    return {r: p for r, p in out.items() if p}
@lru_cache(None)
def ch_coxeter(k):
    out = {}
    for j in range(k):
        sh = tuple([k - j] + [1] * j); sp = schur_p(sh)
        for r, pl in sp.items():
            out[r] = padd(out.get(r, {}), pmul(pl, {k - 1 - j: Fraction((-1) ** j)}))
    return out
@lru_cache(None)
def ch_min(rho):
    A = {(): ONE}
    for k in rho: A = sf_mul(A, ch_coxeter(k))
    return A
def to_h(A):
    """p-basis -> h-basis: coefficient of h_lambda in p_rho is phi^lambda(rho)."""
    out = defaultdict(dict)
    for rho, pl in A.items():
        for lam in parts_n:
            c = Gt[pidx[tuple(lam)]][pidx[tuple(rho)]]
            if c: out[tuple(lam)] = padd(out[tuple(lam)], pmul(pl, {0: Fraction(c)}))
    return {l: p for l, p in out.items() if p}
chT = {}
for v in perms:
    A = {}
    for rho, poly in classpoly(v):
        pl = {k: Fraction(c) for k, c in poly}
        for r, p2 in ch_min(rho).items(): A[r] = padd(A.get(r, {}), pmul(pl, p2))
    chT[v] = A
def ch_graded(w):
    A = {}
    for v in perms:
        P_ = kl.p(v, w)
        if not P_: continue
        pl = {k: Fraction(c) for k, c in enumerate(P_) if c}
        for r, p2 in chT[v].items(): A[r] = padd(A.get(r, {}), pmul(pl, p2))
    return to_h(A)

# --- q = 1 data
f = FastKL(n); Tb = f.tables; Pn = np.array(Tb.perms, dtype=np.int64)
sg = np.array([sgnp(p) for p in Tb.perms], dtype=np.int64)
aut = lambda mu: int(np.prod([factorial(c) for c in Counter([x for x in mu if x >= 2]).values()])) if mu else 1
vals = {}                                  # (i, mu) -> array over w (FastKL order)
for i in range(0, n):
    for mu in (partitions(i) if i else [()]):
        mu = tuple(mu)
        G = all_G(Pn, np.array([n - i] + [y for y in mu if y >= 2], dtype=np.int64), sg, (-1) ** (n - 1 - len(mu)))
        E = evaluate_on_kl_basis(f, G)[0]
        assert (E % aut(mu) == 0).all()
        vals[(i, mu)] = E // aut(mu)

def has(p, pat):
    k = len(pat)
    return any(all((p[I[a]] < p[I[b]]) == (pat[a] < pat[b]) for a in range(k) for b in range(a + 1, k))
               for I in itertools.combinations(range(n), k))
q1check = Counter(); res = Counter(); haiman_q = Counter(); fails = []; uniq = Counter()
for r, w in enumerate(Tb.perms):
    w = tuple(w); H = ch_graded(w); lw = L[w]
    # q = 1 check and Haiman q-positivity
    for li, lam in enumerate(parts_n):
        at1 = sum(H.get(tuple(lam), {}).values())
        tot1 = sum(vals[(i, mu)][r] * (n - i) for (i, mu) in vals if tuple(sorted(mu + (n - i,), reverse=True)) == tuple(lam))
        q1check[at1 == tot1] += 1
    hq = all(c >= 0 and c.denominator == 1 for p in H.values() for c in p.values())
    haiman_q[hq] += 1
    # ILP
    keys = list(vals.keys()); D = lw + 1
    var = {(key, d): t for t, (key, d) in enumerate(itertools.product(keys, range(D)))}
    rows = []; rhs = []
    # main identity: coefficient of h_lambda q^e
    for lam in map(tuple, parts_n):
        target = H.get(lam, {})
        for e in range(0, D + n):
            row = {}
            for (i, mu) in keys:
                a = n - i
                if tuple(sorted(mu + (a,), reverse=True)) != lam: continue
                for d in range(D):
                    for t in range(a):              # [a]_q = 1 + q + ... + q^{a-1}
                        if d + t == e: row[var[((i, mu), d)]] = row.get(var[((i, mu), d)], 0) + 1
            if row or target.get(e, 0):
                rows.append(row); rhs.append(float(target.get(e, 0)))
    # q = 1 values
    for key in keys:
        rows.append({var[(key, d)]: 1 for d in range(D)}); rhs.append(float(vals[key][r]))
    # palindromicity: c(q) = q^{2 cen} c(1/q), cen = (lw - (n-i-1))/2
    if PAL:
        for (i, mu) in keys:
            twice = lw - (n - i - 1)
            for d in range(D):
                d2 = twice - d
                if d2 < 0 or d2 >= D:
                    rows.append({var[((i, mu), d)]: 1}); rhs.append(0.0)
                elif d < d2:
                    rows.append({var[((i, mu), d)]: 1, var[((i, mu), d2)]: -1}); rhs.append(0.0)
    A = np.zeros((len(rows), len(var)))
    for k, row in enumerate(rows):
        for j, c in row.items(): A[k, j] = c
    sol = milp(c=np.zeros(len(var)), constraints=LinearConstraint(A, rhs, rhs), integrality=np.ones(len(var)),
               bounds=Bounds(0, np.inf))
    ok = sol.status == 0
    kind = 'codominant' if not has(w, (2, 0, 1)) else ('singular' if has(w, (2, 3, 0, 1)) or has(w, (3, 1, 2, 0)) else 'smooth')
    res[(kind, ok)] += 1
    if not ok and len(fails) < 5: fails.append(''.join(str(x + 1) for x in w))
    if ok:
        # uniqueness: try to find a different solution by maximising / minimising a random objective
        cvec = np.random.default_rng(r).normal(size=len(var))
        s1 = milp(c=cvec, constraints=LinearConstraint(A, rhs, rhs), integrality=np.ones(len(var)), bounds=Bounds(0, np.inf))
        s2 = milp(c=-cvec, constraints=LinearConstraint(A, rhs, rhs), integrality=np.ones(len(var)), bounds=Bounds(0, np.inf))
        uniq['unique' if np.allclose(s1.x, s2.x) else 'not unique'] += 1
print(f"n={n}: graded character at q=1 equals sum_a a h_a (vertex-1 pieces): {dict(q1check)}")
print(f"n={n} {'(palindromic)' if PAL else ''}: Haiman q-positive graded characters: {dict(haiman_q)}")
print(f"  graded lift of the vertex-1 pieces exists: {dict(res)}; failures {fails}; lifts {dict(uniq)}")
