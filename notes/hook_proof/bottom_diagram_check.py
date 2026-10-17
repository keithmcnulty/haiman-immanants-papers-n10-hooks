"""Exact check of the diagram structure behind the bottom-terms lemma (Lemma 4.6 of the
hooks paper), computed from Kazhdan-Lusztig polynomials, independently of the
Rhoades-Skandera colouring rule.

For m <= M, every v and every c, D_vc(u) is the coefficient of Imm_u in x_{vc} det x[^v,^c]:
    D_vc(u) = sum_{sigma(v)=c} P_{sigma,u}(1) sgn(phi(sigma)),   sgn(phi(sigma)) = (-1)^{v+c} sgn(sigma),
by the duality d_f = sum_u f(C'_u(1)) Imm_u.  Checks:
 (1) D_vc takes values in {0,1}; its support is {e} (iff c = v) plus #cups(v) * #caps(c) elements,
     where #cups(v) = number of v +- 1 in [m], #caps(c) likewise (the colouring rule's prediction);
 (2) D_vk and D_v,k+1 share exactly #cups(v) non-identity elements (the diagrams with cap {k,k+1});
 (3) for every b >= 2 and good rotation rho for v (anchor l), H_c(e) = h(sigma_c), with sigma_c the coset
     element with phi(sigma_c) = e and h(pi) = F^{(b)}_v(rho pi), equals (-1)^{b+|c-l|+1}[|c-l| >= b-1];
 (4) B(u) = sum_c H_c(e) D_vc(u) lies in {0,1} for u != e, and its support has exactly
     #cups(v) * #{k in {l-b+1, l+b-2} : 0 <= k < k+1 <= m-1} elements (0-indexed here).
Usage: .venv/bin/python notes/hook_proof/bottom_diagram_check.py 6
"""
import sys
sys.path.insert(0, ".")  # run from the repository root
from itertools import permutations

from monimm.kazhdan_lusztig import KazhdanLusztig, bruhat_leq, poly_at_one


def sgn(p):
    return (-1) ** sum(1 for a in range(len(p)) for b in range(a + 1, len(p)) if p[a] > p[b])


def cyclen(p, i):
    L, x = 1, p[i]
    while x != i:
        x, L = p[x], L + 1
    return L


def F(b, v, p):
    return (-1) ** (b - 1) * sgn(p) * (1 if cyclen(p, v) >= b else 0)


def rot(m, lo, hi, d):
    z = list(range(m))
    if lo == hi:
        return tuple(z)
    if d == 'down':                       # rho(lo) = hi, rho(x) = x - 1 on (lo, hi]: anchor lo, v = hi
        z[lo] = hi
        z[lo + 1:hi + 1] = range(lo, hi)
    else:                                 # rho(hi) = lo, rho(x) = x + 1 on [lo, hi): anchor hi, v = lo
        z[hi] = lo
        z[lo:hi] = range(lo + 1, hi + 1)
    return tuple(z)


def check(m):
    kl = KazhdanLusztig(m)
    perms = list(permutations(range(m)))
    e = tuple(range(m))
    P1 = {}
    def p1(x, w):
        if (x, w) not in P1:
            P1[x, w] = poly_at_one(kl.p(x, w)) if bruhat_leq(x, w) else 0
        return P1[x, w]
    stats = dict(D=0, shared=0, H=0, B=0)
    for v in range(m):
        cups = sum(1 for y in (v - 1, v + 1) if 0 <= y < m)
        D = {}
        for c in range(m):
            coset = [s for s in perms if s[v] == c]
            Dc = {}
            for u in perms:
                val = sum(p1(s, u) * sgn(s) for s in coset) * (-1) ** (v + c)
                if val:
                    Dc[u] = val
            caps = sum(1 for y in (c - 1, c + 1) if 0 <= y < m)
            assert set(Dc.values()) <= {1}, (m, v, c)
            assert (e in Dc) == (c == v) and len(Dc) - (c == v) == cups * caps, (m, v, c, len(Dc))
            D[c] = Dc
            stats['D'] += 1
        for k in range(m - 1):
            shared = (set(D[k]) & set(D[k + 1])) - {e}
            assert len(shared) == cups, (m, v, k)
            stats['shared'] += 1
        configs = [('id', v, v, v)] + [('down', lo, v, lo) for lo in range(v)] + \
                  [('up', v, hi, hi) for hi in range(v + 1, m)]
        for b in range(2, m + 1):
            for d, lo, hi, l in configs:
                rho = rot(m, lo, hi, d)
                H = {}
                for c in range(m):
                    rest = [x for x in range(m) if x != c]
                    sigma = tuple(c if i == v else rest[i - (i > v)] for i in range(m))
                    H[c] = F(b, v, tuple(rho[sigma[i]] for i in range(m)))
                    pred = (-1) ** (b + abs(c - l) + 1) * (1 if abs(c - l) >= b - 1 else 0)
                    assert H[c] == pred, (m, v, b, d, lo, hi, c)
                    stats['H'] += 1
                B = {}
                for c in range(m):
                    for u, val in D[c].items():
                        B[u] = B.get(u, 0) + H[c] * val
                B.pop(e, None)
                supp = {u for u, val in B.items() if val}
                assert all(B[u] in (0, 1) for u in B), (m, v, b, d, lo, hi)
                ks = [k for k in (l - b + 1, l + b - 2) if 0 <= k and k + 1 <= m - 1]
                assert len(supp) == cups * len(ks), (m, v, b, d, lo, hi, len(supp), ks)
                stats['B'] += 1
    return stats


if __name__ == '__main__':
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    for m in range(2, M + 1):
        print(m, check(m), flush=True)
    print('all checks passed')
