"""Haiman's q-conjecture engine (monimm.kl_graded) against induced Hecke modules built from the relations."""
import os
import itertools
from fractions import Fraction
import numpy as np
from monimm.combinatorics import partitions
from monimm.kazhdan_lusztig import KazhdanLusztig, length
from monimm.kl_extremal import ExtremalKL
from monimm.kl_graded import hecke_traces, c_polynomials, check_properties

def direct_phi_traces(n, q):
    """phi^lambda_q(T_v) at integer q, from the induced modules M^mu (Hecke relations only)."""
    parts = [tuple(p) for p in partitions(n)]
    perms = list(itertools.permutations(range(n)))
    def lmul(i, p): return tuple(i+1 if x == i else i if x == i+1 else x for x in p)
    eta = {}
    for mu in parts:
        inner = set(); pos = 0
        for b in mu:
            inner |= set(range(pos, pos + b - 1)); pos += b
        Dm = [d for d in perms if all(d[i] < d[i+1] for i in inner)]
        idx = {d: k for k, d in enumerate(Dm)}; m = len(Dm)
        Ms = []
        for i in range(n - 1):
            M = np.zeros((m, m), dtype=object)
            for d in Dm:
                sd = lmul(i, d)
                if length(sd) > length(d):
                    if sd in idx: M[idx[sd], idx[d]] += 1
                    else: M[idx[d], idx[d]] += q
                else:
                    M[idx[d], idx[d]] += q - 1; M[idx[sd], idx[d]] += q
            Ms.append(M)
        mat = {tuple(range(n)): np.identity(m, dtype=object)}
        for v in sorted(perms, key=length):
            if v in mat: continue
            for i in range(n - 1):
                sv = lmul(i, v)
                if length(sv) < length(v):
                    mat[v] = Ms[i].dot(mat[sv]); break
        eta[mu] = {v: int(np.trace(mat[v])) for v in perms}
    # h_mu = sum_lambda N[mu][lambda] m_lambda, N = #matrices with row sums mu, col sums lambda
    def count(rows, cols):
        if not rows: return 1 if all(c == 0 for c in cols) else 0
        r, rest = rows[0], rows[1:]; tot = 0
        def rec(j, left, cur):
            nonlocal tot
            if j == len(cols):
                if left == 0: tot += count(rest, tuple(c - x for c, x in zip(cols, cur)))
                return
            for x in range(min(left, cols[j]) + 1): rec(j + 1, left - x, cur + [x])
        rec(0, r, []); return tot
    import sympy
    Nm = sympy.Matrix([[count(mu, lam) for lam in parts] for mu in parts])
    Ninv = Nm.inv()           # m_lambda = sum_mu Ninv[lambda, mu] h_mu
    phi = {lam: {v: sum(Ninv[a, b] * eta[mu][v] for b, mu in enumerate(parts)) for v in perms}
           for a, lam in enumerate(parts)}
    return parts, perms, phi



def test_graded_against_induced_modules():
    for n in range(2, 5):
        K = ExtremalKL(n); T = K.tables
        Tr = hecke_traces(K); C = c_polynomials(K, Tr)
        kl = KazhdanLusztig(n)
        for q in (2, 3):
            parts, perms, phi = direct_phi_traces(n, q)
            assert [tuple(p) for p in T.parts] == parts
            qp = q ** np.arange(Tr.shape[2], dtype=object)
            for v in perms:
                for a, lam in enumerate(parts):
                    assert int(Tr[T.rank(v), a].astype(object).dot(qp)) == phi[lam][v]
            for w in perms:
                for a, lam in enumerate(parts):
                    tot = sum(sum(c * q ** k for k, c in enumerate(kl.p(v, w))) * phi[lam][v]
                              for v in perms if kl.p(v, w))
                    assert int(C[a, T.rank(w)].astype(object).dot(qp)) == tot
        Cs, scale = K.c_coefficients_scaled()
        assert (C.sum(axis=2) * scale == Cs).all()
        props = check_properties(C, T.length)
        assert props["negative_coefficients"] == props["asymmetric"] == props["non_unimodal"] == 0


def test_depth_first_induced_traces():
    """monimm.induced_traces (the n = 10 kernel of check 12) against the induced modules above, every mu, n <= 5:
    phi^lambda_q(T_v) = sum_mu Ninv[lambda, mu] eta^mu_q(T_v) mod p, for every tree cut."""
    import sympy
    from check_graded_induced import _count
    from monimm.induced_traces import eta_traces_all
    p = 2147483629
    for n in range(2, 6):
        for q in (2, 3):
            parts, perms, phi = direct_phi_traces(n, q)
            assert perms == sorted(perms)                      # lexicographic, as eta_traces_all
            Ninv = sympy.Matrix([[_count(mu, lam) for lam in parts] for mu in parts]).inv()
            for cut in (1, 4):
                eta = [eta_traces_all(n, mu, q, p, cut=cut) for mu in parts]
                for a, lam in enumerate(parts):
                    got = sum(int(Ninv[a, b]) * eta[b].astype(object) for b in range(len(parts))) % p
                    assert list(got) == [phi[lam][v] % p for v in perms]


def test_coxeter3_extension_of_extremal_pairs():
    """Check 12 at n = 10 takes P_{v,w} from Coxeter3's extremal pairs, extended by P_{v,w} = P_{v*,w} with v* the
    maximum of W_{D_L(w)} v W_{D_R(w)}.  Against the standard recursion for every pair of S_n, n <= 6."""
    import math
    import pytest
    import check_induced_traces as cit
    from cross_check_coxeter3 import COXETER
    if not os.access(COXETER, os.X_OK):
        pytest.skip("Coxeter3 not available")
    for n in range(2, 7):
        perms = cit.lex_perms(n)
        fact = np.array([math.factorial(k) for k in range(n + 1)], np.int64)
        ws = [tuple(int(x) for x in p) for p in perms]
        cols = cit.kl_columns(n, ws, perms, fact)
        kl = KazhdanLusztig(n)
        for w, c in zip(ws, cols):
            for i, q in enumerate(cit.TRACE_QS):
                want = np.zeros(len(ws), np.int64)
                for x in kl.below[w]:
                    pc = kl.p(x, w)
                    if pc:
                        want[cit.rank(x)] = sum(int(a) * pow(q, e, cit.P) for e, a in enumerate(pc)) % cit.P
                assert (want == c[i]).all(), (n, w, q)
