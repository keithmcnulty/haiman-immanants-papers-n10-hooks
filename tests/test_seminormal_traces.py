"""Hoefsmit's seminormal form (monimm.seminormal_traces) and its use in check 12."""

import numpy as np
import pytest

from monimm.combinatorics import partitions
from monimm.induced_traces import eta_traces_all
from monimm.ordering import kostka
from monimm.seminormal_traces import check_relations, chi_traces_all, standard_tableaux
from run_kl_graded import TRACE_PRIME as P

import check_seminormal_traces


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_relations(n):
    for lam in partitions(n):
        for q in (2, 3):
            ok, msg = check_relations(tuple(lam), q, P)
            assert ok, (lam, q, msg)


def test_dimensions():
    # sum of (f^lambda)^2 = n!, and the largest f^lambda at n = 10 is 768
    assert sum(len(standard_tableaux(tuple(l))) ** 2 for l in partitions(7)) == 5040
    assert max(len(standard_tableaux(tuple(l))) for l in partitions(10)) == 768


def test_trivial_and_sign():
    # chi^{(n)}(T_v) = q^{l(v)} and chi^{(1^n)}(T_v) = (-1)^{l(v)}, v in lexicographic order
    import itertools
    n, q = 5, 3
    perms = list(itertools.permutations(range(n)))
    inv = [sum(1 for i in range(n) for j in range(i + 1, n) if v[i] > v[j]) for v in perms]
    assert list(chi_traces_all(n, (n,), q, P)) == [pow(q, l, P) for l in inv]
    assert list(chi_traces_all(n, (1,) * n, q, P)) == [1 if l % 2 == 0 else P - 1 for l in inv]


def test_induced_modules_decompose():
    # eta^mu_q = sum_lambda K_{lambda mu} chi^lambda_q, independently of the graded engine
    n, q = 6, 2
    parts = [tuple(l) for l in partitions(n)]
    chi = {lam: chi_traces_all(n, lam, q, P) for lam in parts}
    for mu in [(6,), (4, 2), (3, 2, 1), (2, 2, 1, 1)]:
        want = sum(kostka(lam, mu) * chi[lam] for lam in parts) % P
        assert (eta_traces_all(n, mu, q, P) == want).all(), mu


def test_engine_traces_n6(tmp_path):
    assert check_seminormal_traces.main(["--n", "6", "--fresh", "--relations-nmax", "4",
                                         "--out-npz", str(tmp_path / "x.npz")]) == 0


def test_negative_control(tmp_path):
    assert check_seminormal_traces.main(["--n", "5", "--fresh", "--corrupt", "--relations-nmax", "3",
                                         "--out-npz", str(tmp_path / "x.npz")]) == 1
