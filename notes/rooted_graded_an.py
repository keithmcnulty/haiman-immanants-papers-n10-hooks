"""Codominant comparison: the palindromic graded lift of notes/rooted_graded.py vs Abreu-Nigro's graded g_{m,i}(x;q)
(HHKKO arXiv:2504.09123, formula (2.2)): [h_mu] omega g_{m,i}(q) equals our c_{i,mu}(q) up to a power of q, for every
codominant u, n <= 5 (2026-10-14).  Usage: .venv/bin/python notes/rooted_graded_an.py N
"""
# Graded comparison at codominant u: our unique palindromic lift c_{i,mu}(q) vs [h_mu] omega g_{m,i}(x;q) (HHKKO (2.2)).
import sys, itertools, numpy as np
from fractions import Fraction
from collections import Counter, defaultdict
from scipy.optimize import milp, LinearConstraint, Bounds
src = open('notes/rooted_graded.py').read()
exec(src.split("def has(p, pat):")[0])

# symmetric functions in the h basis: dict lam(sorted tuple) -> poly
def hmul(A, B):
    out = defaultdict(dict)
    for la, pa in A.items():
        for lb, pb in B.items():
            l = tuple(sorted(la + lb, reverse=True)); out[l] = padd(out[l], pmul(pa, pb))
    return {l: p for l, p in out.items() if p}
def h_(k): return {(k,) if k else (): ONE}
def e_(k):
    out = defaultdict(dict)
    if k == 0: return {(): ONE}
    for m in range(1, k + 1):
        for comp_ in itertools.product(range(1, k + 1), repeat=m):
            if sum(comp_) != k: continue
            l = tuple(sorted(comp_, reverse=True)); out[l] = padd(out[l], {0: Fraction((-1) ** (k - m))})
    return {l: p for l, p in out.items() if p}
def qint(k): return {d: Fraction(1) for d in range(k)}
rho = {0: {(): ONE}}
for k in range(1, n + 1):
    A = {(k,): qint(k)}
    for j in range(1, k): A = {l: p for l, p in ((l, padd(A.get(l, {}), p2, -1)) for l, p2 in hmul(h_(k - j), rho[j]).items())} | {l: p for l, p in A.items() if l not in hmul(h_(k - j), rho[j])}
    rho[k] = {l: p for l, p in A.items() if p}

def omega_g(m, k):
    """h-expansion of omega(g_{m,k}) = sum (-1)^{|t1|-n+k} q^{wt} e_{|t1|-n+k} rho_{rest}."""
    out = defaultdict(dict)
    for sig in itertools.permutations(range(n)):
        if any(sig[i] > m[i] - 1 for i in range(n)): continue
        seen = set(); cyc = []
        for i in range(n):
            if i in seen: continue
            c = [i]; seen.add(i); j = sig[i]
            while j != i: c.append(j); seen.add(j); j = sig[j]
            cyc.append(c)                      # each cycle starts at its minimum; cycles by increasing minima
        t1 = len(cyc[0]); j_ = t1 - n + k
        if j_ < 0: continue
        word = [x for c in cyc for x in c]; pos = {x: t for t, x in enumerate(word)}
        wt = sum(1 for i in range(n) for jj in range(i + 1, m[i]) if pos[jj] < pos[i])
        A = e_(j_)
        for c in cyc[1:]: A = hmul(A, rho[len(c)])
        for l, p in A.items(): out[l] = padd(out[l], pmul(p, {wt: Fraction((-1) ** j_)}))
    return {l: p for l, p in out.items() if p}

def has(p, pat):
    k = len(pat)
    return any(all((p[I[a]] < p[I[b]]) == (pat[a] < pat[b]) for a in range(k) for b in range(a + 1, k))
               for I in itertools.combinations(range(n), k))
res = Counter(); ex = []
for r, w in enumerate(Tb.perms):
    w = tuple(w)
    if has(w, (2, 0, 1)): continue
    m = [max(w[:i + 1]) + 1 for i in range(n)]
    # our palindromic lift (re-solve)
    H = ch_graded(w); lw = L[w]; keys = list(vals.keys()); D = lw + 1
    var = {(key, d): t for t, (key, d) in enumerate(itertools.product(keys, range(D)))}
    rows = []; rhs = []
    for lam in map(tuple, parts_n):
        target = H.get(lam, {})
        for e in range(0, D + n):
            row = {}
            for (i, mu) in keys:
                a = n - i
                if tuple(sorted(mu + (a,), reverse=True)) != lam: continue
                for d in range(D):
                    for t in range(a):
                        if d + t == e: row[var[((i, mu), d)]] = row.get(var[((i, mu), d)], 0) + 1
            if row or target.get(e, 0): rows.append(row); rhs.append(float(target.get(e, 0)))
    for key in keys: rows.append({var[(key, d)]: 1 for d in range(D)}); rhs.append(float(vals[key][r]))
    for (i, mu) in keys:
        twice = lw - (n - i - 1)
        for d in range(D):
            d2 = twice - d
            if d2 < 0 or d2 >= D: rows.append({var[((i, mu), d)]: 1}); rhs.append(0.0)
            elif d < d2: rows.append({var[((i, mu), d)]: 1, var[((i, mu), d2)]: -1}); rhs.append(0.0)
    A = np.zeros((len(rows), len(var)))
    for k_, row in enumerate(rows):
        for j, c in row.items(): A[k_, j] = c
    sol = milp(c=np.zeros(len(var)), constraints=LinearConstraint(A, rhs, rhs), integrality=np.ones(len(var)), bounds=Bounds(0, np.inf))
    ours = {key: {d: int(round(sol.x[var[(key, d)]])) for d in range(D) if round(sol.x[var[(key, d)]])} for key in keys}
    for i in range(n):
        og = omega_g(m, i)
        for mu in (partitions(i) if i else [()]):
            mu = tuple(mu)
            theirs = {d: int(c) for d, c in og.get(mu, {}).items()}
            mine = ours[(i, mu)]
            # compare up to a q-shift
            if not theirs and not mine: res['both zero'] += 1; continue
            shifts = set()
            if theirs and mine and len(theirs) == len(mine):
                s = min(mine) - min(theirs)
                if all(mine.get(d + s, 0) == c for d, c in theirs.items()): shifts.add(s)
            res['equal up to q-shift' if shifts else 'DIFFERENT'] += 1
            if not shifts and len(ex) < 6: ex.append((''.join(str(x + 1) for x in w), i, mu, mine, theirs))
print(f"n={n}: codominant u, our graded lift vs [h_mu] omega g_(m,i)(q):", dict(res))
for e in ex: print('   ', e)
