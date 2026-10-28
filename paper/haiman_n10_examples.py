"""Exact checks of every number in the worked examples and small figures of paper/haiman_n10.tex (the graded version of paper II).

Run from the repository root:  .venv/bin/python paper/haiman_n10_examples.py   (under a minute)

Each block asserts what the paper prints. Permutations are printed 1-indexed in one-line
notation, as in the paper; internally they are 0-indexed tuples. Left multiplication s_i w swaps
the values i, i+1 and right multiplication w s_i swaps the positions i, i+1.

Blocks:
  Ex 2.A   graded: c_{., 3412}(q) = (1+2q+2q^2+2q^3+q^4, q(1+q)^2, q(1+q)^2, 0, 0), symmetric and unimodal
           about q^2, and equal to (8, 4, 4, 0, 0) at q = 1.
  Ex 5.A   w = 3412: the seven (W_I, W_J)-double cosets of S_4 as contingency tables, their sizes,
           maximal elements and corner sums, and the four extremal pairs (Lemma 5.3).
  Fig 4    r_w(2,3) = min(r_w(1,3) + 1, r_w(3,3)) for w = 3412 (proof of Lemma 5.3).
  Ex 6.A   the recursion (2) for w = 3412, s = s_2, v = 2413: M is empty, v has one extremal
           pair, and P_{1324,3412} = 1 + q, the other three extremal P equal to 1.
  Ex 6.B   the double-coset sums G_lambda and c_{., 3412} = (8, 4, 4, 0, 0).
  Ex 4.A   stability: c_{(3),231} = c_{(3,1),2314} = 3, c_{(2,1),231} = c_{(2,1,1),2314} = 1.
  Ex 8.A   Kostka criterion for (3,1) and (2,2) at n = 4, and N_a(J_mu) values.
  Fig 5    the n = 6 order restricted to (4,2), (3,3), (3,2,1), (2,2,2), (4,1,1), (3,1^3), (2^2,1^2).
  Ex 8.B   Pate pair (3,2) -> (2,2,1): Kostka rows and the criterion.
  Ex 8.C   the J_mu values of the inequality of Example 8.C (already printed in the paper; table form).
  Ex 9.A   n = 3: each w has a vertex of the Newton polytope supported on {w} alone, with an
           integer separating functional.
  Ex 6.C   S_3 traces: ch(T_e), ch(T_{s1}), ch(T_{s1 s2}) in the h basis, the table of phi^lambda_q(T_v),
           the rule tau(T_321) = (q-1) tau(T_312) + q tau(T_132), c_{., 231}(q) = (1+q+q^2, q, 0) and
           c_{., 321}(q) = ([3]_q!, 0, 0).
  Sec 8.1  the q-cones: (1-q+q^2) phi^(2)_q is in C_KL(q) but not in C_J(q) (n = 2), and
           (1-q+q^2) phi^(2,1^{n-2})_q is in C_J^ev(q) but not in C_KL(q) (n = 3, 4, 5; the value at
           231 x e_{n-3} is q - q^2 + q^3).
"""

from __future__ import annotations

import itertools
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monimm.combinatorics import cycle_type, monomial_class_functions, partitions  # noqa: E402
from monimm.kazhdan_lusztig import bruhat_leq, kl_coefficients, length  # noqa: E402
from monimm.ordering import compare, dimension, factorial_of, kostka  # noqa: E402


def P(s):
    return tuple(int(c) - 1 for c in s)


def S(x):
    return "".join(str(v + 1) for v in x)


def d_left(w):
    return [i for i in range(len(w) - 1) if w.index(i) > w.index(i + 1)]


def d_right(w):
    return [i for i in range(len(w) - 1) if w[i] > w[i + 1]]


def blocks(desc, n):
    """Consecutive blocks of {0..n-1} cut by a descent set (i in desc joins i and i+1)."""
    out, cur = [], [0]
    for i in range(n - 1):
        if i in desc:
            cur.append(i + 1)
        else:
            out.append(cur)
            cur = [i + 1]
    out.append(cur)
    return out


def table(y, pos_blocks, val_blocks):
    return tuple(tuple(sum(1 for i in A if y[i] in B) for B in val_blocks) for A in pos_blocks)


def corner(T):
    """S_T(a,b) = sum_{a' <= a, b' >= b} T_{a'b'} (1-indexed a, b in the paper)."""
    r, c = len(T), len(T[0])
    return tuple(tuple(sum(T[a2][b2] for a2 in range(a + 1) for b2 in range(b, c)) for b in range(c))
                 for a in range(r))


def rank(y, i, j):
    """r_y(i,j) = #{i' <= i : y(i') >= j}, 1-indexed i, j."""
    return sum(1 for k in range(i) if y[k] + 1 >= j)


def main():
    kl, co = kl_coefficients(4)
    parts4 = partitions(4)
    _, g = monomial_class_functions(4)
    phi = {lam: dict(zip(parts4, row)) for lam, row in zip(parts4, g)}

    # ---- Ex 5.A -------------------------------------------------------------------------
    w = P("3412")
    I, J = d_left(w), d_right(w)
    assert I == [1] and J == [1]                       # I = J = {s_2}
    alpha, beta = blocks(J, 4), blocks(I, 4)
    assert [len(b) for b in alpha] == [1, 2, 1] and [len(b) for b in beta] == [1, 2, 1]
    cosets = {}
    for y in itertools.permutations(range(4)):
        cosets.setdefault(table(y, alpha, beta), []).append(y)
    rows = []
    for T, ys in cosets.items():
        m = max(ys, key=length)
        assert sum(1 for y in ys if length(y) == length(m)) == 1
        size = 1 * 2 * 1 * 1 * 2 * 1
        for row in T:
            for t in row:
                size //= [1, 1, 2][t]
        assert size == len(ys)
        rows.append((S(m), T, len(ys), corner(T), bruhat_leq(m, w)))
    expected = {  # max element: (size, extremal for w)
        "1324": (2, True), "1432": (4, True), "3214": (4, True), "3412": (4, True),
        "3421": (4, False), "4312": (4, False), "4321": (2, False)}
    assert {r[0]: (r[2], r[4]) for r in rows} == expected
    Sw = corner(table(w, alpha, beta))
    for name, T, size, C, ext in rows:
        dominated = all(C[a][b] <= Sw[a][b] for a in range(3) for b in range(3))
        assert dominated == ext                        # Lemma 5.3
    assert sum(1 for x in kl.perms if bruhat_leq(x, w)) == 14
    print("Ex 5.A  max   table             size  corner sums (a=1,2; b=2,3)  <= w")
    for name, T, size, C, ext in sorted(rows, key=lambda r: (-r[4], r[0])):
        print(f"        {name}  {T}  {size}     {[[C[a][b] for b in (1, 2)] for a in (0, 1)]}"
              f"        {ext}")

    # ---- Fig 4 --------------------------------------------------------------------------
    assert rank(w, 2, 3) == 2 and rank(w, 1, 3) == 1 and rank(w, 3, 3) == 2
    assert rank(w, 2, 3) == min(rank(w, 1, 3) + 1, rank(w, 3, 3))

    # ---- Ex 6.A -------------------------------------------------------------------------
    v = P("2413")
    s = 1
    assert tuple({1: 2, 2: 1}.get(t, t) for t in w) == v
    M = [z for z in kl.perms if z != v and bruhat_leq(z, v) and kl.mu(z, v) != 0
         and z.index(s) > z.index(s + 1)]
    assert M == []
    ext_v = [x for x in kl.perms if bruhat_leq(x, v) and set(d_left(x)) >= set(d_left(v))
             and set(d_right(x)) >= set(d_right(v))]
    assert ext_v == [v]
    assert d_left(v) == [0, 2] and d_right(v) == [1]
    want = {"1324": (1, 1), "1432": (1,), "3214": (1,), "3412": (1,)}
    for name, Pxw in want.items():
        x = P(name)
        sx = tuple({1: 2, 2: 1}.get(t, t) for t in x)
        assert kl.p(x, w) == Pxw
        assert kl.p(sx, v) == (1,)
        assert kl.p(x, v) == ((1,) if name == "1324" else ())
    assert S(tuple({1: 2, 2: 1}.get(t, t) for t in P("1324"))) == "1234"
    # the hat of 1432 for the descents of v is 2431, which is not below v: a failed lookup
    assert not bruhat_leq(P("2431"), v) and length(P("2431")) == 4 > length(v) == 3

    # ---- Ex 6.B -------------------------------------------------------------------------
    Gs = {}
    for name in want:
        ys = cosets[table(P(name), alpha, beta)]
        Gs[name] = tuple(sum(phi[lam][cycle_type(y)] for y in ys) for lam in parts4)
    assert Gs == {"1432": (0, 6, 0, -2, 0), "3214": (0, 6, 0, -2, 0),
                  "3412": (8, -8, 4, 0, 0), "1324": (0, 0, 0, 2, 0)}
    c = tuple(sum(sum(kl.p(P(n_), w)) * Gs[n_][k] for n_ in want) for k in range(5))
    assert c == (8, 4, 4, 0, 0)
    assert tuple(co[lam][w] for lam in parts4) == c
    print("Ex 6.B  G:", Gs, " c:", c)

    # ---- Ex 4.A (stability) --------------------------------------------------------------
    _, co3 = kl_coefficients(3)
    assert co3[(3,)][P("231")] == co[(3, 1)][P("2314")] == 3
    assert co3[(2, 1)][P("231")] == co[(2, 1, 1)][P("2314")] == 1
    assert co3[(1, 1, 1)][P("231")] == co[(1, 1, 1, 1)][P("2314")] == 0
    assert co[(2, 2)][P("2314")] == 0          # (2,2) is not of the form lambda u 1

    # ---- Ex 8.A (Kostka) -----------------------------------------------------------------
    a, b = (3, 1), (2, 2)
    assert (dimension(a), dimension(b)) == (3, 2)
    assert [kostka(a, m) for m in parts4] == [0, 1, 1, 2, 3]
    assert [kostka(b, m) for m in parts4] == [0, 0, 1, 1, 2]
    N = lambda lam, mu: Fraction(factorial_of(mu) * kostka(lam, mu), dimension(lam))
    assert (N(a, (2, 2)), N(b, (2, 2))) == (Fraction(4, 3), 2)
    assert (N(a, (3, 1)), N(b, (3, 1))) == (2, 0)
    assert compare(a, b, True)["witness"] == (2, 2) and compare(b, a, True)["witness"] == (3, 1)

    # ---- Fig 5 (n = 6 zoom) ---------------------------------------------------------------
    names = [(4, 2), (3, 3), (4, 1, 1), (3, 2, 1), (2, 2, 2), (3, 1, 1, 1), (2, 2, 1, 1)]
    rel = {(x, y) for x in names for y in names if x != y and compare(x, y, True)["status"] != "false"}
    covers = {(x, y) for (x, y) in rel if not any((x, z) in rel and (z, y) in rel for z in names)}
    assert covers == {((4, 2), (4, 1, 1)), ((4, 2), (3, 2, 1)), ((4, 2), (2, 2, 2)),
                      ((3, 3), (3, 2, 1)), ((3, 3), (2, 2, 2)),
                      ((4, 1, 1), (3, 1, 1, 1)), ((3, 2, 1), (3, 1, 1, 1)),
                      ((3, 2, 1), (2, 2, 1, 1)), ((2, 2, 2), (2, 2, 1, 1)),
                      ((4, 1, 1), (2, 2, 1, 1))}, covers
    # (4,2) and (3,3) are incomparable, as are (3,2,1) and (2,2,2); (3,3) is not above (4,1,1)
    for x, y in [((4, 2), (3, 3)), ((3, 2, 1), (2, 2, 2)), ((3, 3), (4, 1, 1))]:
        assert (x, y) not in rel and (y, x) not in rel
    # the common lower bounds of (4,2) and (3,3) in all of P(6), and their maximal elements
    p6 = partitions(6)
    low = [z for z in p6 if compare((4, 2), z, True)["status"] != "false"
           and compare((3, 3), z, True)["status"] != "false" and z not in [(4, 2), (3, 3)]]
    maxl = [z for z in low if not any(z != y and compare(y, z, True)["status"] != "false" for y in low)]
    assert sorted(maxl) == [(2, 2, 2), (3, 2, 1)]

    # ---- Ex 8.B (Pate) --------------------------------------------------------------------
    parts5 = partitions(5)
    lam, mu = (3, 2), (2, 2, 1)
    assert dimension(lam) == dimension(mu) == 5
    assert [kostka(lam, m) for m in parts5] == [0, 0, 1, 1, 2, 3, 5]
    assert [kostka(mu, m) for m in parts5] == [0, 0, 0, 0, 1, 2, 5]
    assert [kostka(lam, m) - kostka(mu, m) for m in parts5] == [0, 0, 1, 1, 1, 1, 0]

    # ---- Ex 8.C table -----------------------------------------------------------------------
    p6order = [(6,), (5, 1), (4, 2), (4, 1, 1), (3, 3), (3, 2, 1), (3, 1, 1, 1), (2, 2, 2),
               (2, 2, 1, 1), (2, 1, 1, 1, 1), (1,) * 6]
    vals = []
    for m in p6order:
        # ten times (8.1): 10 per + 20 det - Imm_{chi^3111} - 2 Imm_{chi^222} - 2 Imm_{chi^21111}
        x = (10 * kostka((6,), m) + 20 * kostka((1,) * 6, m) - kostka((3, 1, 1, 1), m)
             - 2 * kostka((2, 2, 2), m) - 2 * kostka((2, 1, 1, 1, 1), m))
        vals.append(x)
    assert vals == [10, 10, 10, 10, 10, 10, 9, 8, 7, 0, 0], vals

    # ---- Ex 9.A (n = 3 isolation) ----------------------------------------------------------
    from monimm.tnn_cone import KLChartData, standard_chart
    import numpy as np
    data = KLChartData(3)
    chart = standard_chart(3)
    assert chart == [("L", 0), ("L", 1), ("L", 0), ("U", 0), ("U", 1), ("U", 0)]
    K, E = data.chart_coefficients(chart)
    E = E.astype(int)
    assert len(E) == 14
    # w -> (exponent vector in (t1,t2,t3,u1,u2,u3), separating functional)
    table9 = {"123": ((0, 0, 0, 0, 0, 0), (-1, -1, -1, -1, -1, -1)),
              "132": ((0, 1, 0, 0, 1, 0), (-1, 1, -1, -1, 1, -1)),
              "213": ((0, 0, 1, 0, 0, 1), (-1, -1, 1, -1, -1, 1)),
              "231": ((0, 1, 1, 0, 1, 1), (-1, 1, 0, -1, 1, 1)),
              "312": ((1, 1, 0, 1, 1, 0), (1, 1, -1, 0, 1, -1)),
              "321": ((0, 1, 2, 1, 1, 1), (-1, 1, 1, 0, 1, 1))}
    idx = {tuple(e): r for r, e in enumerate(E)}
    for name, (alpha_, cvec) in table9.items():
        r = idx[alpha_]
        j = data.perms.index(P(name))
        assert K[r, j] == 1 and (K[r] != 0).sum() == 1          # kappa_alpha = e_w
        vals = E @ np.array(cvec)
        assert all(vals[r] > vals[s2] for s2 in range(len(E)) if s2 != r)   # strict vertex
    # Ex 2.A: graded coefficients for w = 3412
    from monimm.kl_extremal import ExtremalKL
    from monimm.kl_graded import c_polynomials
    KE = ExtremalKL(4); CQ = c_polynomials(KE)
    parts4 = [tuple(p) for p in KE.tables.parts]
    perms4 = [tuple(int(x) for x in p) for p in KE.tables.perms]
    col = perms4.index(P("3412"))
    want = {(4,): [1, 2, 2, 2, 1], (3, 1): [0, 1, 2, 1], (2, 2): [0, 1, 2, 1], (2, 1, 1): [], (1, 1, 1, 1): []}
    for lam, poly in want.items():
        got = [int(x) for x in CQ[parts4.index(lam), col]]
        while got and got[-1] == 0: got.pop()
        assert got == poly, (lam, got)
        if poly:
            lo = next(i for i, x in enumerate(poly) if x)
            assert lo + len(poly) - 1 == 4                       # symmetric about q^{l(w)/2} = q^2
            assert poly[lo:] == poly[lo:][::-1]
            half = poly[lo:lo + (len(poly) - lo + 1) // 2]
            assert all(a <= b for a, b in zip(half, half[1:]))   # unimodal
    assert [sum(want[l]) for l in [(4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1)]] == [8, 4, 4, 0, 0]

    # Ex 6.C: traces in S_3 (polynomials as coefficient lists, lowest degree first)
    from monimm.kl_graded import hecke_traces, minimal_class_values

    def trim(c):
        c = [int(x) for x in c]
        while c and c[-1] == 0:
            c.pop()
        return c

    K3 = ExtremalKL(3); Tr3 = hecke_traces(K3); C3 = c_polynomials(K3, Tr3)
    parts3 = [tuple(p) for p in K3.tables.parts]
    perms3 = [tuple(int(x) for x in p) for p in K3.tables.perms]
    assert parts3 == [(3,), (2, 1), (1, 1, 1)]
    B3 = minimal_class_values(3)            # B[rho, lambda] = phi^lambda_q(T_{c_rho}): the h-coefficients of ch
    hco = {(1, 1, 1): {(1, 1, 1): [1]},
           (2, 1): {(2, 1): [1, 1], (1, 1, 1): [-1]},
           (3,): {(3,): [1, 1, 1], (2, 1): [-2, -1], (1, 1, 1): [1]}}
    for rho, row in hco.items():
        for lam in parts3:
            assert trim(B3[parts3.index(rho), parts3.index(lam)]) == row.get(lam, []), (rho, lam)
    table6 = {"123": [[], [], [1]], "213": [[], [1, 1], [-1]], "132": [[], [1, 1], [-1]],
              "231": [[1, 1, 1], [-2, -1], [1]], "312": [[1, 1, 1], [-2, -1], [1]], "321": [[-1, 0, 0, 1], [2], [-1]]}
    for v, rows in table6.items():
        for li in range(3):
            assert trim(Tr3[perms3.index(P(v)), li]) == rows[li], (v, li)
    # tau(T_321) = (q-1) tau(T_312) + q tau(T_132), coefficientwise for each lambda
    for li in range(3):
        a = [int(x) for x in Tr3[perms3.index(P("312")), li]]
        b = [int(x) for x in Tr3[perms3.index(P("132")), li]]
        rhs = [0] * (len(a) + 1)
        for k, x in enumerate(a):
            rhs[k] -= x; rhs[k + 1] += x
        for k, x in enumerate(b):
            rhs[k + 1] += x
        assert trim(rhs) == trim(Tr3[perms3.index(P("321")), li])
    assert [trim(C3[li, perms3.index(P("231"))]) for li in range(3)] == [[1, 1, 1], [0, 1], []]
    assert [trim(C3[li, perms3.index(P("321"))]) for li in range(3)] == [[1, 2, 2, 1], [], []]
    assert [sum(trim(C3[li, perms3.index(P("231"))])) for li in range(3)] == [3, 1, 0]

    # Sec 8.1: the q-cones
    def pmul(a, b):
        out = [0] * (len(a) + len(b) - 1) if a and b else []
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                out[i + j] += x * y
        return trim(out)
    a_ = [1, -1, 1]                                            # 1 - q + q^2, not in N[q]
    K2 = ExtremalKL(2); C2 = c_polynomials(K2)
    parts2 = [tuple(p) for p in K2.tables.parts]
    l2 = parts2.index((2,))
    vals2 = [pmul(a_, trim(C2[l2, w])) for w in range(C2.shape[1])]
    assert sorted(vals2) == [[], [1, 0, 0, 1]]                  # 0 and 1 + q^3: in C_KL(q), not in C_J(q)
    for n in (3, 4, 5):
        Kn = ExtremalKL(n); Cn = c_polynomials(Kn)
        partsn = [tuple(p) for p in Kn.tables.parts]
        permsn = [tuple(int(x) for x in p) for p in Kn.tables.perms]
        lam = (2,) + (1,) * (n - 2)
        li = partsn.index(lam)
        # evaluations at the w_0(mu), mu a partition: only mu = lam is non-zero, value (1-q+q^2)[2]_q = 1 + q^3
        for mu in partsn:
            w0mu, pos = [], 0
            for m in mu:
                w0mu += list(range(pos + m - 1, pos - 1, -1)); pos += m
            ev = pmul(a_, trim(Cn[li, permsn.index(tuple(w0mu))]))
            assert ev == ([1, 0, 0, 1] if mu == lam else []), (n, mu, ev)
        w = P("231") + tuple(range(3, n))
        assert trim(Cn[li, permsn.index(w)]) == [0, 1]           # c = q (stability)
        assert pmul(a_, [0, 1]) == [0, 1, -1, 1]                 # q - q^2 + q^3: not in C_KL(q)
    print("all checks passed")


if __name__ == "__main__":
    main()
