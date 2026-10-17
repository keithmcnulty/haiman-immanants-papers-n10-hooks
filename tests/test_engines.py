"""Tests of the engines against brute force and against the identities used in the papers.

Run with:  .venv/bin/python -m pytest -q
"""

from fractions import Fraction
from itertools import combinations, permutations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # top-level drivers

from monimm.combinatorics import (cycle_type, known_status, monomial_class_functions,
                                  partition_factorial, partitions, psi_value, z_value)
from monimm.evaluate import ImmanantEvaluator
from monimm.tnn import (build_tnn_batch, build_tnn_exact, j_matrix, lower_word,
                        min_minor, num_params)

# p(n) for n = 1..7.
PARTITION_COUNTS = [1, 2, 3, 5, 7, 11, 15]


def random_tnn(n, rng, batch=1, scale=2.0):
    """Random TP matrices from strictly positive bidiagonal parameters."""
    lower = np.exp(rng.uniform(-scale, scale, (batch, num_params(n))))
    upper = np.exp(rng.uniform(-scale, scale, (batch, num_params(n))))
    return build_tnn_batch(lower, upper)


def brute_force_class_function(phi_of_type, mat):
    """d_phi(A) straight from the definition, summing over all of S_n."""
    n = len(mat)
    total = 0.0
    for perm in permutations(range(n)):
        total += phi_of_type(cycle_type(perm)) * np.prod([mat[i][perm[i]] for i in range(n)])
    return total


# -- combinatorics ---------------------------------------------------------

@pytest.mark.parametrize("n", range(1, 8))
def test_partition_counts(n):
    assert len(partitions(n)) == PARTITION_COUNTS[n - 1]


@pytest.mark.parametrize("n", range(1, 8))
def test_class_sizes_sum_to_factorial(n):
    # sum_rho n!/z(rho) = n!  <=>  sum_rho 1/z(rho) = 1.
    assert sum(Fraction(1, z_value(rho)) for rho in partitions(n)) == 1


@pytest.mark.parametrize("n", range(1, 7))
def test_psi_matches_coset_count(n):
    # psi^mu(sigma) = number of cosets of S_mu fixed by sigma, checked by brute force:
    # cosets of S_mu <-> functions {0..n-1} -> blocks with block sizes mu.
    from itertools import product
    for mu in partitions(n):
        labelings = [lab for lab in product(range(len(mu)), repeat=n)
                     if all(lab.count(j) == mu[j] for j in range(len(mu)))]
        for rho in partitions(n):
            # A representative permutation of cycle type rho.
            perm, start = [], 0
            for length in rho:
                perm.extend(list(range(start + 1, start + length)) + [start])
                start += length
            fixed = sum(all(lab[perm[i]] == lab[i] for i in range(n)) for lab in labelings)
            assert psi_value(mu, rho) == fixed


@pytest.mark.parametrize("n", range(1, 8))
def test_dual_basis(n):
    # <g_lambda, psi^mu> = delta_{lambda mu}  (the definition of the monomial immanants).
    parts, g = monomial_class_functions(n)
    for a, lam in enumerate(parts):
        for mu in parts:
            inner = sum(g[a][r] * psi_value(mu, rho) / z_value(rho)
                        for r, rho in enumerate(parts))
            assert inner == (1 if lam == mu else 0)


def test_known_status_n6():
    # For n = 6 exactly (4,2) and (3,2,1) are not covered by the published theorems.
    assert [mu for mu in partitions(6) if not known_status(mu)] == [(4, 2), (3, 2, 1)]


def test_known_status_counts():
    # Table 2, last column: shapes outside S_n for n = 6, 7.
    assert sum(1 for mu in partitions(7) if not known_status(mu)) == 6


# -- TNN matrices ----------------------------------------------------------

@pytest.mark.parametrize("n", range(2, 8))
def test_j_matrices_are_tnn(n):
    # Every minor of J_mu is 0 or 1, so J_mu is TNN (Lemma 6.1 of the hooks paper).
    for mu in partitions(n):
        assert min_minor(j_matrix(mu)) > -1e-12


@pytest.mark.parametrize("n", range(2, 6))
def test_factorization_gives_tp(n):
    # Positive bidiagonal parameters give a totally positive matrix.
    rng = np.random.default_rng(n)
    for mat in random_tnn(n, rng, batch=5, scale=1.0):
        assert min_minor(mat) > 0


@pytest.mark.parametrize("n", range(2, 8))
def test_word_is_reduced_for_w0(n):
    # The word must have length n(n-1)/2 and its product of simple transpositions
    # must be the longest element w0 (i -> n-1-i).
    word = lower_word(n)
    assert len(word) == num_params(n)
    perm = list(range(n))
    for k in word:
        perm[k], perm[k + 1] = perm[k + 1], perm[k]
    assert perm == list(range(n))[::-1]


def test_exact_and_float_construction_agree():
    rng = np.random.default_rng(1)
    lower = np.exp(rng.uniform(-1, 1, num_params(4)))
    upper = np.exp(rng.uniform(-1, 1, num_params(4)))
    diag = np.exp(rng.uniform(-1, 1, 4))
    approx = build_tnn_batch(lower[None], upper[None], diag[None])[0]
    exact = build_tnn_exact(lower, upper, diag)
    assert np.allclose(approx, np.array(exact, dtype=float))


# -- immanant evaluation ---------------------------------------------------

@pytest.mark.parametrize("n", range(2, 6))
def test_batched_matches_brute_force(n):
    ev = ImmanantEvaluator(n)
    rng = np.random.default_rng(n)
    mat = random_tnn(n, rng)[0]
    fast = ev.monomial(mat)[0]
    for a in range(len(ev.parts)):
        g_row = dict(zip(ev.parts, ev.g_exact[a]))
        slow = brute_force_class_function(lambda rho: float(g_row[rho]), mat)
        assert fast[a] == pytest.approx(slow, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("n", range(2, 8))
def test_sum_of_monomials_is_permanent(n):
    # per = sum_lambda Imm_lambda, since sum_lambda g_lambda = 1 (pair with psi^(n) = 1).
    ev = ImmanantEvaluator(n)
    mats = random_tnn(n, np.random.default_rng(n), batch=4)
    assert np.allclose(ev.monomial(mats).sum(axis=1), ev.permanent(mats))


@pytest.mark.parametrize("n", range(2, 6))
def test_lemma_0_3_expansion(n):
    # d_phi(A) = sum_lambda d_phi(J_lambda)/lambda! * Imm_lambda(A), for a random
    # integer class function phi and a random (not necessarily TNN) matrix A.
    ev = ImmanantEvaluator(n)
    rng = np.random.default_rng(10 + n)
    phi = [Fraction(int(v)) for v in rng.integers(-5, 6, len(ev.parts))]
    mat = [[Fraction(int(v)) for v in row] for row in rng.integers(-3, 4, (n, n))]
    lhs = ev.class_function_exact(phi, mat)
    imm = ev.monomial_exact(mat)
    rhs = sum(ev.class_function_exact(phi, [[Fraction(int(v)) for v in row]
                                           for row in j_matrix(lam)])
              / partition_factorial(lam) * imm[a]
              for a, lam in enumerate(ev.parts))
    assert lhs == rhs


@pytest.mark.parametrize("n", range(2, 7))
def test_monotonicity_in_n(n):
    # Stability: Imm_{lambda u 1}(A (+) [1]) = Imm_lambda(A).
    small, big = ImmanantEvaluator(n - 1), ImmanantEvaluator(n)
    mat = random_tnn(n - 1, np.random.default_rng(n))[0]
    padded = np.zeros((n, n))
    padded[:n - 1, :n - 1] = mat
    padded[n - 1, n - 1] = 1.0
    small_vals, big_vals = small.monomial(mat)[0], big.monomial(padded)[0]
    for a, lam in enumerate(small.parts):
        assert big_vals[big.index(lam + (1,))] == pytest.approx(small_vals[a], rel=1e-9)


@pytest.mark.parametrize("n", range(2, 8))
def test_determinant_and_permanent_rows(n):
    # det = d_{chi^(1^n)} = sum_lambda K_{(1^n),lambda} Imm_lambda (Young's rule),
    # and K_{(1^n),lambda} = delta_{lambda,(1^n)}, so Imm_{(1^n)} = det.
    ev = ImmanantEvaluator(n)
    mat = random_tnn(n, np.random.default_rng(n))[0]
    det_index = ev.index((1,) * n)
    # det(A) = 1 here (unitriangular factors) while per(A) can be ~1e14, so the
    # float value suffers cancellation; it must agree within the rounding bound.
    sums = ev.class_sums(mat)
    value = ev.monomial_from_sums(sums)[0][det_index]
    bound = ev.error_bound(sums)[0][det_index]
    assert abs(value - np.linalg.det(mat)) <= bound
    # The exact computation gives det = 1 on the nose.
    exact = ev.monomial_exact(build_tnn_exact(*[np.exp(np.random.default_rng(n).uniform(
        -2, 2, num_params(n))) for _ in range(2)]))
    assert exact[det_index] == 1


# -- Kazhdan-Lusztig immanants -------------------------------------------------

from monimm.kazhdan_lusztig import (KazhdanLusztig, kl_coefficients, kl_immanant_coefficients,
                                    length, longest_times)


def one_indexed(text):
    return tuple(int(ch) - 1 for ch in text)


def test_known_kl_polynomials_in_s4():
    """In S_4 the only non-trivial P_{x,w} are 1 + q, for w = 3412 or 4231."""
    kl = KazhdanLusztig(4)
    assert kl.p(one_indexed("1324"), one_indexed("3412")) == (1, 1)
    assert kl.p(one_indexed("2143"), one_indexed("4231")) == (1, 1)
    assert kl.p(one_indexed("1234"), one_indexed("4321")) == (1,)
    nontrivial = {w for (x, w), p in kl.P.items() if p != (1,)}
    assert nontrivial == {one_indexed("3412"), one_indexed("4231")}


@pytest.mark.parametrize("n", range(1, 6))
def test_kl_inversion_formula(n):
    """sum_z (-1)^{l(x)+l(z)} P_{x,z}(q) P_{w0 w, w0 z}(q) = delta_{x,w}."""
    kl = KazhdanLusztig(n)

    def mul(a, b):
        out = [0] * max(0, len(a) + len(b) - 1)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                out[i + j] += x * y
        return out

    for w in kl.perms:
        for x in kl.below[w]:
            total = [0]
            for z in kl.below[w]:
                term = mul(kl.p(x, z), kl.p(longest_times(w), longest_times(z)))
                sign = (-1) ** (length(x) + length(z))
                total = [a + sign * b for a, b in
                         zip(total + [0] * (len(term) - len(total)), term + [0] * (len(total) - len(term)))]
            assert [c for c in total if c] == ([1] if x == w else [])


@pytest.mark.parametrize("n", range(1, 6))
def test_extreme_kl_immanants(n):
    """Imm_e = det and Imm_{w0} = x_{1,n} x_{2,n-1} ... x_{n,1}."""
    kl = KazhdanLusztig(n)
    identity, w0 = tuple(range(n)), tuple(range(n))[::-1]
    assert kl_immanant_coefficients(kl, identity) == {v: (-1) ** length(v) for v in kl.perms}
    assert kl_immanant_coefficients(kl, w0) == {w0: 1}


@pytest.mark.parametrize("n", range(1, 6))
def test_monomial_immanants_have_nonnegative_kl_coefficients(n):
    """kl_coefficients checks the identity itself; here, c_{lambda,w} >= 0 for n <= 5."""
    _, coeffs = kl_coefficients(n)
    assert all(c >= 0 for row in coeffs.values() for c in row.values())
    # Imm_{1^n} = det = Imm_e.
    assert {w: c for w, c in coeffs[(1,) * n].items() if c} == {tuple(range(n)): 1}


# ---------------------------------------------------------------------------
# Immanant ordering via the matrices J_mu (monimm/ordering.py)
# ---------------------------------------------------------------------------

from monimm.ordering import (character, compare, dimension, inner_with_psi, kostka,
                             proved_order)
from math import factorial

from monimm.combinatorics import partition_factorial


@pytest.mark.parametrize("n", range(1, 7))
def test_kostka_against_characters(n):
    parts = partitions(n)
    assert all(inner_with_psi(a, mu) == kostka(a, mu) for a in parts for mu in parts)
    assert sum(dimension(a) ** 2 for a in parts) == factorial(n)


def _character_immanants(evaluator, mat, n):
    """d_{chi^a}(A) from characters and class sums (independent of Kostka numbers)."""
    sums = evaluator.class_sums_exact(mat)
    return {a: sum(character(a, rho) * s for rho, s in zip(evaluator.parts, sums))
            for a in partitions(n)}


@pytest.mark.parametrize("n", range(2, 7))
def test_character_immanants_at_j_matrices(n):
    """d_{chi^a}(J_mu) = mu! K_{a,mu}: Lemma 3.1 of the n <= 10 paper, with Young's rule."""
    evaluator = ImmanantEvaluator(n)
    for mu in partitions(n):
        values = _character_immanants(evaluator, j_matrix(mu).tolist(), n)
        assert all(values[a] == partition_factorial(mu) * kostka(a, mu) for a in values)


@pytest.mark.parametrize("n", range(2, 7))
def test_proved_relations_hold_on_tp_matrices(n):
    """Every proved relation holds at exact totally positive matrices, and every
    'false' relation fails at its J_mu witness."""
    rng = np.random.default_rng(n)
    evaluator = ImmanantEvaluator(n)
    for normalized in (False, True):
        rel = proved_order(n, normalized)
        for _ in range(3):
            k = num_params(n)
            mat = build_tnn_exact(rng.integers(1, 50, k).tolist(), rng.integers(1, 50, k).tolist())
            values = _character_immanants(evaluator, mat, n)
            for (a, b), v in rel.items():
                if v["status"].startswith("proved"):
                    lhs, rhs = values[a], values[b]
                    if normalized:
                        lhs, rhs = lhs * dimension(b), rhs * dimension(a)
                    assert lhs >= rhs, (a, b, normalized)
        for (a, b), v in rel.items():
            if v["status"] == "false":
                values = _character_immanants(evaluator, j_matrix(v["witness"]).tolist(), n)
                lhs, rhs = values[a], values[b]
                if normalized:
                    lhs, rhs = lhs * dimension(b), rhs * dimension(a)
                assert lhs < rhs, (a, b, normalized)


# ---------------------------------------------------------------------------
# Compiled KL engine (monimm/kl_fast.py)
# ---------------------------------------------------------------------------

from monimm.kl_fast import FastKL


@pytest.mark.parametrize("n", range(1, 7))
def test_fast_kl_matches_reference(n):
    """Every P_{x,w} and every c_{lambda,w} equal the dict-based implementation."""
    fast = FastKL(n)
    kl, ref = kl_coefficients(n)
    assert fast.npairs == len(kl.P)
    assert all(fast.p(x, w) == kl.p(x, w) for w in kl.perms for x in kl.perms)
    c = fast.c_coefficients()            # includes the exact identity check
    assert all(c[lam][w] == ref[lam][w] for lam in ref for w in kl.perms)
    assert fast.sanity() == {"constant_term_one": True, "nonnegative_coefficients": True}


# ---------------------------------------------------------------------------
# Hooks: the (3,1^{n-3}) decomposition and Conjecture 10.1 of the hooks paper
# ---------------------------------------------------------------------------

from monimm.kl_fast import evaluate_on_kl_basis, vertex_hook_table


@pytest.mark.parametrize("n", range(3, 7))
def test_hook31_decomposition(n):
    """f^(2)_i is the 0/1 indicator of {s_(i-1), s_i, s_(i-1)s_i, s_i s_(i-1)} (Lemma);
    f^(3)_i = sum_j B_ij - f^(2)_i (identity (4)), with every B_ij >= 0."""
    fast = FastKL(n)
    T = fast.tables
    perms = np.array(T.perms)
    sign = np.where(T.length % 2 == 0, 1, -1)
    F = vertex_hook_table(fast)
    simple = lambda k: tuple(k + 1 if x == k else k if x == k + 1 else x for x in range(n))
    compose = lambda u, v: tuple(u[v[x]] for x in range(n))
    for i in range(n):
        expected = set()
        for k in (i - 1, i):
            if 0 <= k < n - 1:
                expected.add(simple(k))
        if 1 <= i <= n - 2:
            expected |= {compose(simple(i - 1), simple(i)), compose(simple(i), simple(i - 1))}
        support = {T.perms[w] for w in np.nonzero(F[:, 1, i])[0]}
        assert support == expected and set(F[:, 1, i].tolist()) <= {0, 1}
        others = [j for j in range(n) if j != i]
        B = evaluate_on_kl_basis(fast, np.array(
            [-sign * ((perms[:, i] == j) & (perms[:, j] == i)) for j in others]))
        assert B.min() >= 0
        assert np.array_equal(B.sum(axis=0) - F[:, 1, i], F[:, 2, i])


@pytest.mark.parametrize("n", range(2, 7))
def test_two_cycle_product_rule(n):
    """Conjecture 10.1 (B_ij) equals the computed KL expansion of x_ij x_ji det(rest) for every pair."""
    from check_further_computations import predicted_B
    fast = FastKL(n)
    T = fast.tables
    perms = np.array(T.perms)
    sign = np.where(T.length % 2 == 0, 1, -1)
    for i in range(n):
        for j in range(i + 1, n):
            row = evaluate_on_kl_basis(fast, (-sign * ((perms[:, i] == j) & (perms[:, j] == i)))[None, :])[0]
            actual = {T.perms[r]: int(row[r]) for r in np.nonzero(row)[0]}
            assert actual == predicted_B(n, i, j, T.perms), (i, j)


# ---------------------------------------------------------------------------
# Parallel engine for n = 9 (monimm/kl_big.py)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", range(2, 7))
def test_big_engine_matches_fast_engine(n):
    from monimm.kl_big import BigKL
    big, fast = BigKL(n), FastKL(n)
    assert big.npairs == fast.npairs
    for w in range(big.tables.N):
        bi = slice(big.start[w], big.start[w] + big.size[w])
        fi = slice(fast.start[w], fast.start[w] + fast.size[w])
        assert np.array_equal(big.below[bi], fast.below[fi])
        assert np.array_equal(big.polys[big.pid[bi]], fast.polys[fast.pid[fi]])
    C, scale = big.c_coefficients_scaled()
    fast.c_coefficients()
    assert scale == 1 and np.array_equal(C, fast.C_scaled)


# ---------------------------------------------------------------------------
# Extremal-pair engine for n = 10 (monimm/kl_extremal.py)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", range(2, 7))
def test_extremal_tables_match_brute_force(n):
    """Tables dominated by T(w) = {x <= w : D_L(x) >= D_L(w), D_R(x) >= D_R(w)}."""
    from itertools import permutations
    from monimm.kazhdan_lusztig import bruhat_leq
    from monimm.kl_extremal import ExtremalKL, _descents
    E = ExtremalKL(n)
    T = E.tables
    perms = [tuple(int(a) for a in p) for p in T.perms]
    D = [_descents(np.array(p)) for p in perms]
    for w, pw in enumerate(perms):
        Lw, Rw = D[w]
        brute = sorted(x for x, px in enumerate(perms)
                       if D[x][0] & Lw == Lw and D[x][1] & Rw == Rw and bruhat_leq(px, pw))
        assert list(E.xs[E.start[w]:E.start[w] + E.size[w]]) == brute


@pytest.mark.parametrize("n", range(2, 6))
def test_extremal_engine_polynomials_match_reference(n):
    from itertools import permutations
    from monimm.kl_extremal import ExtremalKL
    E, kl = ExtremalKL(n), KazhdanLusztig(n)
    for w in permutations(range(n)):
        for x in permutations(range(n)):
            ref = list(kl.p(x, w))
            while ref and ref[-1] == 0:
                ref.pop()
            assert E.P(x, w) == ref


@pytest.mark.parametrize("n", range(2, 8))
def test_extremal_engine_matches_big_engine(n):
    from monimm.kl_big import BigKL
    from monimm.kl_extremal import ExtremalKL
    E = ExtremalKL(n)
    C, scale = E.c_coefficients_scaled()
    CB, scaleB = BigKL(n).c_coefficients_scaled()
    assert scale == scaleB and np.array_equal(C, CB)
    assert E.sampled_identity(C, samples=10 ** 9)          # every monomial
    bad = C.copy()
    bad[-1, -1] += 1                                       # the identity must notice
    assert not E.sampled_identity(bad, samples=10 ** 9)
    ex = E.sampled_identity_exact(C, samples=10 ** 9)
    assert ex["exact"] and ex["no_wraparound"]
    exb = E.sampled_identity_exact(bad, samples=10 ** 9)
    assert not exb["exact"] and exb["mismatches_mod_p"] > 0


def test_exact_identity_against_stored_table():
    """The stored-table route of check_inversion_exact.py: G from g_table, C from a big run."""
    from monimm.kl_extremal import ExtremalKL
    E = ExtremalKL(6)
    C, _ = E.c_coefficients_scaled()
    G = E.G.copy()
    E.G = None
    G2, _ = E.g_table()
    assert np.array_equal(G, G2)
    assert E.sampled_identity_exact(C, samples=200, seed=3)["exact"]


# ---------------------------------------------------------------------------
# TNN cone versus KL cone on the full span (monimm/tnn_cone.py)
# ---------------------------------------------------------------------------

def test_tnn_isolation_small_n():
    from monimm.tnn_cone import KLChartData, is_singular, isolated_set, parse_perm
    data3 = KLChartData(3)
    assert isolated_set(data3, directions=20_000) == set(data3.perms)
    data4 = KLChartData(4)
    non = {w for w in data4.perms if w not in isolated_set(data4, directions=20_000)}
    assert non == {parse_perm("3412"), parse_perm("4231")}
    assert non == {w for w in data4.perms if is_singular(w)}


def test_tnn_witnesses_n5_exact():
    """f >= 0 on TNN_5 (non-negative chart coefficients) with a negative KL coefficient."""
    from monimm.tnn_cone import (KLChartData, parse_perm, random_charts, standard_chart,
                                 verify_witness)
    data = KLChartData(5)
    charts = [standard_chart(5)] + list(random_charts(5, 5, 0))
    for chart, combo in ((0, {"15432": 1, "43215": 1, "45312": -1}),
                         (0, {"21543": 1, "54321": 1, "52431": -1}),
                         (5, {"32154": 1, "54321": 1, "53241": -1})):
        assert verify_witness(data, {parse_perm(k): a for k, a in combo.items()}, charts[chart])
    # negative controls: dropping a positive term breaks the certificate
    assert not verify_witness(data, {parse_perm("21543"): 1, parse_perm("52431"): -1}, charts[0])
    assert not verify_witness(data, {parse_perm("54321"): 1}, charts[0])   # KL-positive: not a witness


# ---------------------------------------------------------------------------
# Structure of the immanant orders (monimm/order_structure.py)
# ---------------------------------------------------------------------------

from monimm.order_structure import (dominates, hasse_tikz, height, order_geq, structure, width)
from monimm.ordering import cover_relations


@pytest.mark.parametrize("n", range(1, 9))
def test_order_structure_basics(n):
    """Both orders are partial orders inside dominance, their covers agree with
    ordering.cover_relations, and height/width agree with brute force."""
    parts = partitions(n)
    for normalized in (False, True):
        geq = order_geq(n, normalized)
        assert all(dominates(a, b) for a, b in geq)
        assert not any((b, a) in geq for a, b in geq)                       # antisymmetric
        assert all((a, c) in geq for a, b in geq for b2, c in geq if b == b2 and a != c)
        d = structure(n, normalized)
        assert sorted(d["covers"]) == sorted(cover_relations(n, normalized))
        if n <= 6:
            chains = lambda S: all((a, b) in geq or (b, a) in geq for i, a in enumerate(S) for b in S[i + 1:])
            anti = lambda S: not any((a, b) in geq or (b, a) in geq for i, a in enumerate(S) for b in S[i + 1:])
            subsets = [[p for k, p in enumerate(parts) if m >> k & 1] for m in range(1, 2 ** len(parts))]
            assert d["height"] == max(len(S) for S in subsets if chains(S))
            assert d["width"] == max(len(S) for S in subsets if anti(S))


def test_order_structure_small_facts():
    """Facts read off by hand from Kostka matrices."""
    # n = 3 plain: (3) and (2,1) are incomparable (per(I) = 1 < 2 = Imm_{21}(I); at J_3 0 < 1).
    assert structure(3, False)["maximal"] == [(3,), (2, 1)]
    # n = 4 normalized: everything comparable except (3,1) vs (2,2).
    d = structure(4, True)
    assert d["dominance_pairs_dropped"] == [((3, 1), (2, 2))]
    # normalized hook chain is monotone (Skandera's theorem) for every n <= 10
    assert all(structure(n, True)["chains"]["hooks"]["monotone"] for n in range(2, 11))
    assert "\\draw" in hasse_tikz(5, True)
