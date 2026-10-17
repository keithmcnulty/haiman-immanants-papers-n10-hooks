"""The ordering of irreducible character immanants on TNN matrices, via the matrices J_mu.

Setting
-------
For a real class function phi on S_n, the duality <g_lambda, psi^mu> = delta defining the
monomial immanants and Lemma 6.1 of the hooks paper (Lemma 3.1 of the n <= 10 paper) give

    d_phi = sum_{lambda |- n} <phi, psi^lambda> Imm_lambda,
    d_phi(J_mu) = mu! <phi, psi^mu>.

For phi = chi^a (an irreducible character), <chi^a, psi^lambda> = K_{a,lambda}, the
Kostka number (Young's rule).  So for two immanants and positive weights
alpha, beta

    alpha d_{chi^a} - beta d_{chi^b} = sum_lambda c_lambda Imm_lambda,
    c_lambda = alpha K_{a,lambda} - beta K_{b,lambda}.                    (*)

We use two orderings:

* unnormalized:  Imm_{chi^a} >= Imm_{chi^b}  (alpha = beta = 1);
* normalized:    Imm_{chi^a} / f^a >= Imm_{chi^b} / f^b, f = chi(e) = K_{., 1^n}
  (alpha = f^b, beta = f^a after clearing denominators).  This is the
  classical normalization of Schur, Lieb, Merris, Heyfron and Stembridge.

What (*) proves
---------------
* If some c_mu < 0, the inequality is false on TNN_n: J_mu is TNN and the
  difference equals mu! c_mu < 0 there.  This is unconditional.
* If all c_lambda >= 0, the inequality holds on TNN_n as soon as Imm_lambda >= 0
  on TNN_n for every lambda with c_lambda > 0 (the "support").  It is proved
  when every shape in the support is one where this is known
  (:func:`monimm.combinatorics.known_status`: hooks, at most two columns,
  rectangles, two rows with n <= 5), or when MN_n itself is known.

MN_n for n <= 10 rests on our computer proof that every monomial immanant is a
non-negative combination of Kazhdan-Lusztig immanants
(``results/ext_n10.json`` and the graded run ``results/kl_graded_n10.json``; n = 10
implies every smaller n by the stability lemma c_{lambda u 1^k, w x e_k} = c_{lambda,w}).
Rhoades-Skandera then give Imm_w >= 0 on TNN.  :data:`MN_PROVED_NMAX` records
this.  So for n <= 10 every pair (a, b) is either proved comparable or proved
incomparable, and the full order is determined.  For n >= 11 incomparability is
still proved, but some comparabilities are conditional on open shapes.

Kostka numbers are computed by removing horizontal strips.  Characters, via
Murnaghan-Nakayama, are used only to check them against <chi^a, psi^mu>
(tests).
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import factorial
from typing import Dict, List, Tuple

from .combinatorics import Partition, known_status, partitions, psi_value, z_value

# MN_n is proved for n <= MN_PROVED_NMAX by the KL computation (see docstring).
MN_PROVED_NMAX = 10


# ---------------------------------------------------------------------------
# Kostka numbers and characters
# ---------------------------------------------------------------------------

def _horizontal_strips(shape: Partition, size: int):
    """All shapes nu inside ``shape`` with shape / nu a horizontal strip of ``size`` boxes.

    shape / nu is a horizontal strip iff shape_{i+1} <= nu_i <= shape_i for all i.
    """
    rows = list(shape)

    def build(i, remaining, prefix):
        if i == len(rows):
            if remaining == 0:
                yield tuple(x for x in prefix if x > 0)
            return
        lower = rows[i + 1] if i + 1 < len(rows) else 0
        for nu_i in range(rows[i], lower - 1, -1):
            take = rows[i] - nu_i
            if take > remaining:
                break
            yield from build(i + 1, remaining - take, prefix + [nu_i])

    yield from build(0, size, [])


@lru_cache(maxsize=None)
def kostka(shape: Partition, content: Partition) -> int:
    """K_{shape, content}: semistandard tableaux of the given shape and content.

    The entries equal to the last letter form a horizontal strip, so remove the
    last part of ``content`` as a strip and recurse.  Any order of the content
    gives the same number, so a partition is used throughout.
    """
    if sum(shape) != sum(content):
        return 0
    if not content:
        return 1
    last, rest = content[-1], content[:-1]
    return sum(kostka(nu, rest) for nu in _horizontal_strips(shape, last))


def dimension(shape: Partition) -> int:
    """f^shape = chi^shape(e) = K_{shape, 1^n}."""
    return kostka(shape, (1,) * sum(shape))


@lru_cache(maxsize=None)
def character(shape: Partition, rho: Partition) -> int:
    """chi^shape on the class rho, by Murnaghan-Nakayama (beta-numbers / abacus).

    Removing a border strip of length r is moving a bead from position b to
    b - r on the abacus.  The sign is (-1)^{height}, where the height is the
    number of beads strictly between b - r and b.
    """
    if not rho:
        return 1 if not shape else 0
    r, rest = rho[0], rho[1:]
    length = len(shape)
    beta = [shape[i] + (length - 1 - i) for i in range(length)]   # distinct, decreasing
    beads = set(beta)
    total = 0
    for b in beta:
        target = b - r
        if target < 0 or target in beads:
            continue
        height = sum(1 for x in beads if target < x < b)
        new = sorted((beads - {b}) | {target}, reverse=True)
        nu = tuple(x - (length - 1 - i) for i, x in enumerate(new))
        nu = tuple(p for p in nu if p > 0)
        total += (-1) ** height * character(nu, rest)
    return total


def inner_with_psi(shape: Partition, mu: Partition) -> Fraction:
    """<chi^shape, psi^mu> from characters; equals K_{shape, mu} (Young's rule)."""
    n = sum(shape)
    return sum((Fraction(character(shape, rho) * psi_value(mu, rho), z_value(rho))
                for rho in partitions(n)), Fraction(0))


# ---------------------------------------------------------------------------
# Comparing two immanants
# ---------------------------------------------------------------------------

def difference_coefficients(a: Partition, b: Partition, normalized: bool) -> Dict[Partition, int]:
    """c_lambda in (*) for alpha Imm_{chi^a} - beta Imm_{chi^b} (integers; see docstring)."""
    n = sum(a)
    alpha, beta = (dimension(b), dimension(a)) if normalized else (1, 1)
    return {lam: alpha * kostka(a, lam) - beta * kostka(b, lam) for lam in partitions(n)}


def compare(a: Partition, b: Partition, normalized: bool) -> Dict:
    """Status of "Imm_{chi^a} >= Imm_{chi^b} on TNN_n" (normalized or not).

    Returns a dict with ``status`` one of

    * ``"false"``: some c_mu < 0, so J_mu is a counterexample (``witness``);
    * ``"proved (known shapes)"``: c >= 0, and every shape in the support is known;
    * ``"proved (MN_n)"``: c >= 0, and n <= MN_PROVED_NMAX;
    * ``"conditional"``: c >= 0, but the support contains shapes where MN is open (``open``).
    """
    n = sum(a)
    c = difference_coefficients(a, b, normalized)
    negative = [lam for lam, x in c.items() if x < 0]
    if negative:
        # The most negative coefficient (ties: first in reverse-lex order).
        witness = min(negative, key=lambda lam: c[lam])
        return {"status": "false", "witness": witness, "value_at_J": factorial_of(witness) * c[witness]}
    support = [lam for lam, x in c.items() if x > 0]
    open_shapes = [lam for lam in support if not known_status(lam)]
    if not open_shapes:
        return {"status": "proved (known shapes)", "support": support}
    if n <= MN_PROVED_NMAX:
        return {"status": "proved (MN_n)", "support": support, "open": open_shapes}
    return {"status": "conditional", "support": support, "open": open_shapes}


def factorial_of(mu: Partition) -> int:
    """mu! = |S_mu|."""
    out = 1
    for part in mu:
        out *= factorial(part)
    return out


def order_relations(n: int, normalized: bool) -> Dict[Tuple[Partition, Partition], Dict]:
    """compare(a, b) for all ordered pairs a != b of partitions of n."""
    parts = partitions(n)
    return {(a, b): compare(a, b, normalized) for a in parts for b in parts if a != b}


def cover_relations(n: int, normalized: bool) -> List[Tuple[Partition, Partition]]:
    """Hasse diagram of the order "a >= b" (all c >= 0), as (upper, lower) pairs.

    Only uses whether c >= 0, i.e. the order that holds given MN_n.
    """
    parts = partitions(n)
    geq = {(a, b) for a in parts for b in parts if a != b
           and all(x >= 0 for x in difference_coefficients(a, b, normalized).values())}
    return [(a, b) for (a, b) in geq
            if not any((a, m) in geq and (m, b) in geq for m in parts)]


def hook(k: int, n: int) -> Partition:
    """The hook (k, 1^{n-k})."""
    return (k,) + (1,) * (n - k)


def external_relations(n: int, normalized: bool) -> Dict[Tuple[Partition, Partition], str]:
    """Inequalities on TNN_n proved in the literature by other methods.

    Normalized hook chain chi^{(k,1^{n-k})}/f >= chi^{(k-1,1^{n-k+1})}/f for all n: Skandera,
    arXiv:2510.00327, Theorem 5.1 (the TNN analogue of Merris-Heyfron; preprint).
    """
    if not normalized:
        return {}
    source = "Skandera 2025, Thm 5.1 (hook chain)"
    return {(hook(k, n), hook(k - 1, n)): source for k in range(2, n + 1)}


def proved_order(n: int, normalized: bool) -> Dict[Tuple[Partition, Partition], Dict]:
    """:func:`order_relations` plus external theorems, closed under transitivity.

    A pair that is "conditional" becomes "proved (transitivity)" when
    a >= m >= b with both steps proved, by any route.  The result is recorded
    with a ``via`` chain.  Pairs that are "false" stay false: transitivity
    cannot contradict a J_mu counterexample, and an assertion checks this.
    """
    rel = order_relations(n, normalized)
    for pair, source in external_relations(n, normalized).items():
        if rel[pair]["status"] == "conditional":
            rel[pair] = dict(rel[pair], status="proved (external)", source=source)
    proved = {p for p, v in rel.items() if v["status"].startswith("proved")}
    parts = partitions(n)
    via = {}
    changed = True
    while changed:                         # Warshall-style closure, small p(n)
        changed = False
        for m in parts:
            for a in parts:
                if (a, m) not in proved:
                    continue
                for b in parts:
                    if b != a and (m, b) in proved and (a, b) not in proved:
                        proved.add((a, b))
                        via[(a, b)] = m
                        changed = True
    for pair, m in via.items():
        assert rel[pair]["status"] != "false", f"transitivity contradicts a J_mu witness at {pair}"
        rel[pair] = dict(rel[pair], status="proved (transitivity)", via=m)
    return rel
