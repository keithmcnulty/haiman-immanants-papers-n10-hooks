"""Exact checks of every number in the worked examples and figures of paper/hooks_kl.tex.

Run from the repository root:  .venv/bin/python paper/hooks_kl_examples.py   (a few seconds)

Each block asserts what the paper prints, using the KL engine in monimm/kazhdan_lusztig.py.
Permutations are printed 1-indexed in one-line notation, as in the paper; internally they are
0-indexed tuples.  The expansion of a polynomial sum_v f(v) x^v in the Kazhdan-Lusztig immanants is
sum_w f(C'_w(1)) Imm_w with f(C'_w(1)) = sum_{v <= w} P_{v,w}(1) f(v) (eq. (2) of the paper).

Blocks:
  Ex 3.A   x_12 det x[^1,^2] = Imm_213 + Imm_312, and the table of (P_{213,w}, P_{231,w}).
  Cor 4.2  n = 3: Imm_(2,1) = 2 Imm_213 + 2 Imm_132 + Imm_231 + Imm_312.
  Ex 4.A   closure lemma table for rho = 125346 (down-rotation of [3,5]), b = 2..5.
  Fig 2    the values H_c(e) and the pair sums for m = 6, v = 2, rho = e, b = 3.
  Ex 4.7   the decomposition of f^(3)_1 at n = 4 by the value c = pi(1).
  Ex 7.A   Imm_v(A_2341), Imm_v(A_2143), Imm_v(J_(2,2)).
  Ex 7.B   one clean and one non-clean step at n = 4.
  Fig 4    shaded cells of the 3412 / 4231 configurations are empty for n <= 6.
  Fig 6    tangent-space dimensions at w(2 5) for w = 15432, 15342, and P_{e,15342} = 1 + q.
  Ex 9.A   the lifts for w = 45312 (3412) and w = 52431 (4231).
  Tab 1    the support of B_34 at n = 6 (Conjecture 10.1(2), interior adjacent case).
  Thm 9.2  all three elements have only non-negative coefficients in the stated charts (870, 916, 10,611 monomials,
           coefficients 1..16).
  Prop 9.3 the chart claims: Imm_3412 = m, Imm_4231 = m1 + m2 + m3, the monomials a, b, a_i, b_i and m^2 = ab,
           m_i^2 = a_j b_k.  The b_i are monomials of Imm_4321 = x14 x23 x32 x41, not of det, which is 1 in this chart
           (and Imm_4231^2 <= 3 Imm_2143 det fails on TP_4, e.g. at t_1 = ... = t_12 = 10; asserted).
  Sec 1    Imm_1342, Imm_1423 occur in Imm_(3,1) and are negative on a matrix with minors of order <= 2
           non-negative (Dong-Wu-Gao's weaker hypothesis), so KL-positivity does not give their theorem.
"""

from __future__ import annotations

import bisect
import itertools
import sys
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monimm.combinatorics import cycle_type, monomial_class_functions  # noqa: E402
from monimm.kazhdan_lusztig import (KazhdanLusztig, bruhat_leq,  # noqa: E402
                                    kl_immanant_coefficients, length, poly_at_one)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@lru_cache(None)
def KL(n):
    return KazhdanLusztig(n)


def P1(x, w):
    return poly_at_one(KL(len(w)).p(x, w))


def name(w):
    return ''.join(str(a + 1) for a in w)


def perm(s):
    return tuple(int(c) - 1 for c in s)


def names(d):
    return {name(k): v for k, v in d.items() if v}


def sgn(w):
    return (-1) ** length(w)


def std(seq):
    s = sorted(seq)
    return tuple(s.index(a) for a in seq)


def restrict(sig, I):
    return std([sig[i] for i in I])


def compose(rho, pi):
    return tuple(rho[pi[x]] for x in range(len(pi)))


def expand(f, n):
    """{w: f(C'_w(1))} for the polynomial sum_v f(v) x^v."""
    kl = KL(n)
    out = {}
    for w in kl.perms:
        c = sum(P1(v, w) * f.get(v, 0) for v in kl.below[w])
        if c:
            out[w] = c
    return out


@lru_cache(None)
def immcoef(u):
    """Imm_u as {v: coefficient of x^v}."""
    return kl_immanant_coefficients(KL(len(u)), u)


def product_fn(n, I, T, F, G):
    """sum_v f(v) x^v = Imm_F(x[I,T]) Imm_G(x[I^c,T^c]); F, G are perms or {perm: coefficient}."""
    Ic = [i for i in range(n) if i not in I]
    F = F if isinstance(F, dict) else {F: 1}
    G = G if isinstance(G, dict) else {G: 1}
    f = {}
    for sig in itertools.permutations(range(n)):
        if {sig[i] for i in I} != set(T):
            continue
        a, b = restrict(sig, I), restrict(sig, Ic)
        val = (sum(c * immcoef(u).get(a, 0) for u, c in F.items())
               * sum(c * immcoef(v).get(b, 0) for v, c in G.items()))
        if val:
            f[sig] = val
    return f


def cyclen(s, i):
    L, j = 1, s[i]
    while j != i:
        j, L = s[j], L + 1
    return L


@lru_cache(None)
def F(b, v, n):
    """The per-vertex function F^(b)_v(sigma) = (-1)^(b-1) sgn(sigma) [l_v(sigma) >= b] on S_n."""
    return {s: (-1) ** (b - 1) * sgn(s) * (1 if cyclen(s, v) >= b else 0) for s in KL(n).perms}


def H_c(h, m, v, c):
    """H_c(pi') = h(pi) for pi(v) = c, pi' = pi with position v and value c deleted."""
    return {std([s[i] for i in range(m) if i != v]): h[s] for s in KL(m).perms if s[v] == c}


def down(l, h, m):
    """Down-rotation of the block [l, h] (1-indexed): rho(l) = h, rho(x) = x - 1 for l < x <= h."""
    r = list(range(m))
    r[l - 1] = h - 1
    for x in range(l + 1, h + 1):
        r[x - 1] = x - 2
    return tuple(r)


def up(l, h, m):
    """Up-rotation of the block [l, h] (1-indexed): rho(h) = l, rho(x) = x + 1 for l <= x < h."""
    r = list(range(m))
    r[h - 1] = l - 1
    for x in range(l, h):
        r[x - 1] = x
    return tuple(r)


def c_vector(w):
    """(phi^lambda(C'_w(1)))_lambda in the order of monomial_class_functions."""
    n = len(w)
    parts, g = monomial_class_functions(n)
    return tuple(sum(P1(v, w) * row[parts.index(cycle_type(v))] for v in KL(n).below[w])
                 for row in g)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def check_ex_3A():
    f = product_fn(3, [0], [1], (0,), (0, 1))
    assert names(expand(f, 3)) == {'213': 1, '312': 1}
    table = {name(w): (P1(perm('213'), w), P1(perm('231'), w)) for w in KL(3).perms}
    assert table == {'123': (0, 0), '132': (0, 0), '213': (1, 0), '231': (1, 1),
                     '312': (1, 0), '321': (1, 1)}
    print("Ex 3.A ok")


def check_cor_4_2():
    parts, g = monomial_class_functions(3)
    row = g[parts.index((2, 1))]
    f = {v: row[parts.index(cycle_type(v))] for v in KL(3).perms}
    assert names(expand(f, 3)) == {'213': 2, '132': 2, '231': 1, '312': 1}
    print("Cor 4.2 (n = 3) ok")


def check_ex_4A():
    m, v = 6, 4
    rho = down(3, 5, m)
    assert name(rho) == '125346'
    pred = {1: (up(1, 2, 5), 1), 2: (up(2, 2, 5), 2), 4: (down(3, 3, 5), 3),
            5: (down(3, 4, 5), 4), 6: (down(3, 5, 5), 5)}
    assert [name(pred[c][0]) for c in (1, 2, 4, 5, 6)] == ['21345', '12345', '12345', '12435', '12534']
    assert {c: rho[c - 1] + 1 for c in range(1, 7)} == {1: 1, 2: 2, 3: 5, 4: 3, 5: 4, 6: 6}
    for b in (2, 3, 4, 5):
        h = {p: F(b, v, m)[compose(rho, p)] for p in KL(m).perms}
        for c in range(1, m + 1):
            Hc = H_c(h, m, v, c - 1)
            if c == 3:                                    # the anchor
                assert not any(Hc.values())
                continue
            r, vp = pred[c]
            assert all(Hc[p] == F(b - 1, vp - 1, m - 1)[compose(r, p)] for p in KL(m - 1).perms)
    print("Ex 4.A ok")


def check_fig_2():
    m, v, b = 6, 1, 3
    h = F(b, v, m)                                        # rho = e
    He = [H_c(h, m, v, c)[tuple(range(m - 1))] for c in range(m)]
    assert He == [0, 0, 0, 1, -1, 1]
    assert [He[k] + He[k + 1] for k in range(m - 1)] == [0, 0, 1, 0, 0]
    print("Fig 2 strip ok")


def check_ex_4_7():
    n, b, v = 4, 3, 0
    h = F(b, v, n)
    assert names(expand(h, n)) == {'2314': 1, '3124': 1, '3214': 2, '3241': 1, '4213': 1, '4231': 2}
    expected = {2: {'213': ({'2314': 1, '3214': 1})},
                3: {'123': {'3124': 1, '4123': 1}, '213': {'3214': 1, '4213': 1},
                    '231': {'3241': 1, '4231': 1}},
                4: {'123': {'4123': 1}, '231': {'4231': 1}}}
    hvals = {2: {'213': 1}, 3: {'123': 1, '213': 1, '231': 1}, 4: {'123': -1, '231': 1}}
    rot = {2: ('123', 1), 3: ('213', 2), 4: ('312', 3)}
    total = {}
    for c in range(n):
        Hc = H_c(h, n, v, c)
        if c == 0:
            assert not any(Hc.values())
            continue
        r, vp = rot[c + 1]
        assert all(Hc[p] == F(2, vp - 1, n - 1)[compose(perm(r), p)] for p in KL(n - 1).perms)
        Hexp = expand({k: x for k, x in Hc.items() if x}, n - 1)
        vals = {u: (Hc[u] if u == (0, 1, 2) else Hexp.get(u, 0)) for u in KL(n - 1).perms}
        assert names(vals) == hvals[c + 1]
        for u, val in vals.items():
            if not val:
                continue
            prod = expand(product_fn(n, [v], [c], (0,), u), n)
            assert names(prod) == expected[c + 1][name(u)]
            for w, a in prod.items():
                total[w] = total.get(w, 0) + a * val
    assert names(total) == names(expand(h, n))
    print("Ex 4.7 ok")


def check_ex_7A():
    def A(w):
        mm = [max(w[:i + 1]) + 1 for i in range(len(w))]
        return [[1 if j + 1 <= mm[i] else 0 for j in range(len(w))] for i in range(len(w))]

    def value(u, M):
        return sum(c * eval_mono(M, x) for x, c in immcoef(u).items())

    def eval_mono(M, x):
        p = 1
        for i, j in enumerate(x):
            p *= M[i][j]
        return p
    J22 = [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]
    assert A(perm('2341')) == [[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1]]
    assert A(perm('2143')) == [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]]
    for M, w in ((A(perm('2341')), '2341'), (A(perm('2143')), '2143'), (J22, '2143')):
        assert {name(u): value(u, M) for u in KL(4).perms if value(u, M)} == {w: 1}
    print("Ex 7.A ok")


def check_ex_7B():
    kl = KL(4)
    parts, _ = monomial_class_functions(4)
    assert parts == ((4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1))

    def M_set(w, l):
        return [name(z) for z in kl.below[w] if z != w and z[l] > z[l + 1] and kl.mu(z, w)]
    assert M_set(perm('3142'), 1) == []
    assert c_vector(perm('3142')) == (4, 2, 2, 0, 0) and c_vector(perm('3412')) == (8, 4, 4, 0, 0)
    assert M_set(perm('2341'), 0) == ['2143']
    assert c_vector(perm('2341')) == (4, 2, 2, 0, 0)
    assert c_vector(perm('3241')) == (8, 4, 0, 0, 0)
    assert c_vector(perm('2143')) == (0, 0, 4, 0, 0)
    print("Ex 7.B ok")


def check_fig_4(nmax=6):
    allowed = {'3412': {(1, 3), (3, 1)}, '4231': {(1, 1), (3, 3)}}
    counts = {}
    for n in range(4, nmax + 1):
        kl = KL(n)
        cnt = {'3412': 0, '4231': 0}
        for w in kl.perms:
            for z in kl.below[w]:
                if kl.length[w] - kl.length[z] != 3 or not kl.mu(z, w):
                    continue
                D = [i for i in range(n) if w[i] != z[i]]
                assert len(D) == 4
                pat, zp = name(std([w[i] for i in D])), name(std([z[i] for i in D]))
                assert (pat, zp) in (('3412', '1324'), ('4231', '2143'))
                cnt[pat] += 1
                vals = sorted(w[i] for i in D)
                for k in range(n):
                    if k not in D and D[0] < k < D[3] and vals[0] < w[k] < vals[3]:
                        cell = (bisect.bisect(D, k), bisect.bisect(vals, w[k]))
                        assert cell in allowed[pat], (name(w), name(z), cell)
        counts[n] = cnt
    assert counts.get(4) == {'3412': 1, '4231': 1}
    assert counts.get(5) == {'3412': 18, '4231': 18}
    if nmax >= 6:
        assert counts[6] == {'3412': 243, '4231': 243}
    print("Fig 4 ok", counts)


def check_fig_6():
    def tdim(x, w):
        n = len(w)
        c = 0
        for p in range(n):
            for q in range(p + 1, n):
                y = list(x)
                y[p], y[q] = y[q], y[p]
                c += bruhat_leq(tuple(y), w)
        return c
    for ws, xs, lx, lw, td in (('15432', '12435', 1, 6, 6), ('15342', '12345', 0, 5, 6)):
        w = perm(ws)
        x = list(w)
        x[1], x[4] = x[4], x[1]
        x = tuple(x)
        assert (name(x), length(x), length(w), tdim(x, w)) == (xs, lx, lw, td)
    assert KL(5).p(perm('12345'), perm('15342')) == (1, 1)
    print("Fig 6 ok")


def check_ex_9A():
    n = 5
    for ws, pat, extra in (('45312', '3412', True), ('52431', '4231', False)):
        w, I = perm(ws), [0, 1, 3, 4]
        T = sorted(w[i] for i in I)
        assert name(restrict(w, I)) == pat and restrict(w, [2]) == (0,)
        A = {u: expand(product_fn(n, I, T, u, (0,)), n).get(w, 0) for u in KL(4).perms}
        assert names(A) == {pat: 1}
        if extra:
            assert [t + 1 for t in T] == [1, 2, 4, 5]
            assert P1(perm('15342'), w) == 1 and P1(perm('42315'), w) == 1
            lift = product_fn(n, I, T, {perm('1432'): Fraction(1, 2), perm('3214'): Fraction(1, 2),
                                        perm('3412'): -1}, (0,))
            E = expand(lift, n)
            assert {name(k): x for k, x in E.items() if x < 0} == {'45312': -1}
    print("Ex 9.A ok")


def check_tab_Badj():
    """Table of B_{i,i+1} for n = 6, i = 3: Q = {2,3,4,5}, 18 patterns with value 1, 3412 with 2, four 5-cycles."""
    n, i, j = 6, 2, 3                                  # 0-indexed
    rest = [k for k in range(n) if k not in (i, j)]
    f = {}
    for v in KL(n).perms:
        if v[i] == j and v[j] == i:
            f[v] = sgn(std([v[k] for k in rest]))
    B = names(expand(f, n))
    S4 = list(itertools.permutations(range(4)))
    top = [x for x in S4 if bruhat_leq(perm('1324'), x) and x != perm('4321')]
    pattern = {name((0,) + tuple(a + 1 for a in x) + (5,)): (2 if x == perm('3412') else 1) for x in top}
    assert len(pattern) == 19 and sum(1 for c in pattern.values() if c == 1) == 18
    cycles = {'345126': 1, '451236': 1, '145623': 1, '156234': 1}
    assert B == {**pattern, **cycles} and len(B) == 23
    print("Tab B_{i,i+1} (n = 6, i = 3) ok")


def check_thm_9_2():
    from monimm.tnn_cone import KLChartData, parse_perm
    data = KLChartData(5)
    # L_i(t) = I + t E_{i+1,i}, U_i(t) = I + t E_{i,i+1}; tnn_cone uses 0-indexed letters
    lower = (4, 3, 4, 2, 3, 4, 1, 2, 3, 4)
    standard = [("L", i - 1) for i in lower] + [("U", i - 1) for i in reversed(lower)]
    word = "L1 L2 U1 L3 U3 L1 U2 L2 U4 U3 L1 L4 L3 U1 L2 L1 U2 U4 U1 U3".split()
    chart = [(x[0], int(x[1]) - 1) for x in word]
    assert [i + 1 for k, i in chart if k == "L"] == [1, 2, 3, 1, 2, 1, 4, 3, 2, 1]
    assert [i + 1 for k, i in chart if k == "U"] == [1, 3, 2, 4, 3, 1, 2, 4, 1, 3]
    for ch, combo, size in ((standard, {"15432": 1, "43215": 1, "45312": -1}, 870),
                            (standard, {"21543": 1, "54321": 1, "52431": -1}, 916),
                            (chart, {"32154": 1, "54321": 1, "53241": -1}, 10611)):
        E = {k: v for k, v in data.exact_expansion(ch, {parse_perm(u): c for u, c in combo.items()}).items() if v}
        assert len(E) == size and min(E.values()) == 1 and max(E.values()) <= 16, (combo, len(E))
    print("Thm 9.2 charts ok (870, 916, 10611 monomials, all coefficients in [1, 16])")


def check_prop_9_3():
    from monimm.tnn_cone import KLChartData, parse_perm
    data = KLChartData(4)
    word = (3, 2, 1, 3, 2, 3)
    chart = [("L", i - 1) for i in word] + [("U", i - 1) for i in word]
    E = {u: {k: v for k, v in data.exact_expansion(chart, {parse_perm(u): 1}).items() if v}
         for u in ("1432", "3214", "2143", "4321", "1234", "3412", "4231")}

    def mono(s):                                   # "t2t3^2t9" -> exponent vector in t1..t12
        e = [0] * 12
        for part in s.split("t")[1:]:
            k, _, p = part.partition("^")
            e[int(k) - 1] += int(p or 1)
        return tuple(e)

    add = lambda x, y: tuple(p + q for p, q in zip(x, y))
    dbl = lambda x: tuple(2 * p for p in x)
    for u in ("1432", "3214", "2143", "4321"):
        assert min(E[u].values()) > 0, u
    m = mono("t2t3t4t5t8t9t10t11")
    assert E["3412"] == {m: 1}
    ms = [mono("t1t2t3^2t5t8t9^2t11t12"), mono("t1t2t3^2t9^2t11t12"), mono("t1t2t3t9t11t12")]
    assert E["4231"] == {x: 1 for x in ms}
    a, b = mono("t2t4t5t8t10t11"), mono("t2t3^2t4t5t8t9^2t10t11")
    assert a in E["1432"] and b in E["3214"] and add(a, b) == dbl(m)
    a1, a2 = mono("t1t3t5^2t8^2t9t12"), mono("t1t3t9t12")
    b1, b2 = mono("t1t2^2t3^3t9^3t11^2t12"), mono("t1t2^2t3t9t11^2t12")
    assert {a1, a2} <= set(E["2143"]) and {b1, b2} <= set(E["4321"])
    assert dbl(ms[0]) == add(a1, b1) and dbl(ms[1]) == add(a2, b1) and dbl(ms[2]) == add(a2, b2)
    # det = 1 in a unitriangular chart, so the b_i are not monomials of det, and the inequality with det fails:
    assert E["1234"] == {(0,) * 12: 1}
    ev = lambda e, t: sum(c * t ** sum(k) for k, c in e.items())      # all t_i = t
    assert ev(E["4231"], 10) ** 2 > 3 * ev(E["2143"], 10) * ev(E["1234"], 10)
    assert ev(E["4231"], 10) ** 2 <= 3 * ev(E["2143"], 10) * ev(E["4321"], 10)
    print("Prop 9.3 chart claims ok (Imm_4231^2 <= 3 Imm_2143 Imm_4321; the version with det fails at t = 10)")


def check_dwg_tp2():
    """Sec 1: the KL expansion of Imm_(3,1) does not prove non-negativity on TP_2(4)."""
    A = [[3, 1, 1, 0], [3, 1, 1, 1], [2, 1, 1, 1], [2, 1, 1, 2]]
    AT = [list(r) for r in zip(*A)]

    def det(M):
        if len(M) == 1:
            return M[0][0]
        return sum((-1) ** j * M[0][j] * det([r[:j] + r[j + 1:] for r in M[1:]]) for j in range(len(M)))

    def minors(M, k):
        return [det([[M[i][j] for j in J] for i in I])
                for I in itertools.combinations(range(4), k) for J in itertools.combinations(range(4), k)]

    def imm(u, M):
        total = 0
        for v, c in immcoef(perm(u)).items():
            term = c
            for i in range(4):
                term *= M[i][v[i]]
            total += term
        return total

    assert min(minors(A, 1)) >= 0 and min(minors(A, 2)) >= 0 and min(minors(A, 3)) < 0
    k = monomial_class_functions(4)[0].index((3, 1))
    assert c_vector(perm("1342"))[k] == 3 and c_vector(perm("1423"))[k] == 3
    assert imm("1342", A) == -1 and imm("1423", AT) == -1
    print("Sec 1: Imm_1342(A) = Imm_1423(A^T) = -1 on TP_2(4), coefficient 3 in Imm_(3,1)   ok")


if __name__ == "__main__":
    check_ex_3A()
    check_cor_4_2()
    check_ex_4A()
    check_fig_2()
    check_ex_4_7()
    check_ex_7A()
    check_ex_7B()
    check_fig_6()
    check_ex_9A()
    check_tab_Badj()
    check_thm_9_2()
    check_prop_9_3()
    check_dwg_tp2()
    check_fig_4()
    print("all checks passed")
