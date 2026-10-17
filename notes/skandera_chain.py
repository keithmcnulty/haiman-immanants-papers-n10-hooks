#!/usr/bin/env python3
"""Checks for Corollary 5.1 of the hooks paper: KL-positivity of Skandera's hook chain (arXiv:2510.00327, Thm 5.1).

    .venv/bin/python notes/skandera_chain.py --nmax 12 --klmax 8      # ~1 min

D_k = (k-1) chi^{(k,1^{n-k})} - (n-k+1) chi^{(k-1,1^{n-k+1})}  (k = 2..n), Skandera's (5.2).
Identity (proved in the hooks paper, checked here exactly as class functions):

    D_k = sum_{j=0}^{k-2} h_j * m_{(k-j,1^{n-k})},

where * is the induction product (Frobenius: product of symmetric functions), h_j is the trivial
character of S_j and m_nu = g_nu is the monomial class function (Imm_nu = d_{g_nu}).  On the immanant side
the j-th term is sum_{|T|=j} per(x[T,T]) Imm_{(k-j,1^{n-k})}(x[T^c,T^c]), which is KL-positive by Theorems 1.3
and 3.1 of the hooks paper.  For n <= klmax the script also checks directly that D_k and every
term have non-negative coefficients on the KL basis (evaluate_on_kl_basis).
"""

from __future__ import annotations

import argparse
import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from monimm.combinatorics import cycle_type, monomial_class_functions, partitions, z_value  # noqa: E402
from monimm.ordering import character  # noqa: E402


def hook(k, n):
    return tuple([k] + [1] * (n - k))


def monomial(nu):
    """g_nu as a dict cycle type -> value (rows of monomial_class_functions follow partitions(n))."""
    parts, g = monomial_class_functions(sum(nu))
    row = g[parts.index(nu)]
    return dict(zip(partitions(sum(nu)), row))


def remove_part(t, a):
    r = list(t)
    r.remove(a)
    return tuple(r)


def times_p(a, F, n):
    """Induction product p_a * F on S_n: (p_a F)(tau) = a m_a(tau) F(tau minus a part a)."""
    return {t: (a * t.count(a) * F[remove_part(t, a)] if a in t else Fraction(0)) for t in partitions(n)}


def h_times(j, F, m):
    """h_j * F on S_{m+j}, with h_j = sum_{rho |- j} p_rho / z_rho."""
    out = {t: Fraction(0) for t in partitions(m + j)}
    for rho in (partitions(j) if j else [()]):
        cur, size = F, m
        for a in rho:
            size += a
            cur = times_p(a, cur, size)
        z = z_value(rho) if j else 1
        for t in out:
            out[t] += cur[t] / z
    return out


def chain_difference(k, n):
    return {t: Fraction((k - 1) * character(hook(k, n), t) - (n - k + 1) * character(hook(k - 1, n), t))
            for t in partitions(n)}


def terms(k, n):
    return [h_times(j, monomial(hook(k - j, n - j)), n - j) for j in range(k - 1)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nmax", type=int, default=12)
    ap.add_argument("--klmax", type=int, default=8)
    args = ap.parse_args(argv)
    pairs = 0
    for n in range(2, args.nmax + 1):
        for k in range(2, n + 1):
            T = terms(k, n)
            rhs = {t: sum(v[t] for v in T) for t in partitions(n)}
            assert rhs == chain_difference(k, n), (n, k)
            pairs += 1
    print(f"identity D_k = sum_j h_j * m_(k-j,1^(n-k)) holds for all {pairs} pairs (n, k), n <= {args.nmax}")
    from monimm.kl_fast import FastKL, evaluate_on_kl_basis
    for n in range(3, args.klmax + 1):
        f = FastKL(n)
        types = [cycle_type(tuple(p)) for p in f.tables.perms]
        rows, labels = [], []
        for k in range(2, n + 1):
            for lab, F in [(("D", k), chain_difference(k, n))] + [(("term", k, j), v) for j, v in enumerate(terms(k, n))]:
                assert all(F[t].denominator == 1 for t in F)
                rows.append([int(F[t]) for t in types])
                labels.append(lab)
        E = evaluate_on_kl_basis(f, np.array(rows, dtype=np.int64))
        bad = [labels[i] for i in range(len(rows)) if E[i].min() < 0]
        print(f"n={n}: {len(rows)} functions (D_k and all terms) on the KL basis; negative: {bad or 'none'}", flush=True)
        assert not bad
    return 0


if __name__ == "__main__":
    sys.exit(main())
