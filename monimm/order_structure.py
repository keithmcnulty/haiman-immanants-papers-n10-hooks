"""Structure of the two immanant orders on TNN_n (plain and normalized), for n <= MN_PROVED_NMAX.

The orders
----------
For partitions a, b of n write a >= b if Imm_{chi^a} >= Imm_{chi^b} on TNN_n (plain), or
Imm_{chi^a}/f^a >= Imm_{chi^b}/f^b (normalized).  By the J_mu test (:mod:`monimm.ordering`; Proposition 6.3 of the hooks paper) and MN_n,

    plain:       a >= b  <=>  K_{a,mu} >= K_{b,mu}            for all mu |- n,
    normalized:  a >= b  <=>  f^b K_{a,mu} >= f^a K_{b,mu}    for all mu |- n.

Both are partial orders: reflexive and transitive, and antisymmetric because Kostka rows are
linearly independent (and, normalized, the rows K_{a,.}/f^a are distinct).  These are complete
answers for n <= MN_PROVED_NMAX, the range where MN_n is proved.

Containment in dominance (proved for all n)
------------------------------------------
If a >= b in either order then a dominates b.  Evaluate at mu = b: K_{b,b} = 1, so
K_{a,b} >= 1 (plain) or f^b K_{a,b} >= f^a > 0 (normalized).  Hence K_{a,b} > 0, and
K_{a,b} != 0 iff a dominates b.  So each order is a weakening of dominance, and
reverse-lexicographic order on partitions (a linear extension of dominance) is a linear extension
of both.  :func:`structure` records how far each order is from dominance.

What is computed (:func:`structure`)
----------------------------------
* the number of comparable pairs, the covers, and the pairs of dominance that each order drops;
* whether the poset is a lattice: every pair has a join (least upper bound) and a meet, with
  the first obstruction if not;
* height (number of elements in a longest chain) and width (largest antichain; by Dilworth,
  p(n) minus a maximum matching in the bipartite comparability graph);
* the maximal and minimal elements;
* natural chains: hooks (n) > (n-1,1) > ... > (1^n), two-row shapes (n) > (n-1,1) > ...,
  two-column shapes (2^k,1^{n-2k}), and whether consecutive members are comparable;
* whether conjugation a -> a' reverses the order (an anti-automorphism).

:func:`hasse_tikz` draws the Hasse diagram with each element at the height of the longest chain
below it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

from .combinatorics import Partition, partitions
from .ordering import dimension, kostka

Pair = Tuple[Partition, Partition]


def dominates(a: Partition, b: Partition) -> bool:
    """a dominates b: every partial sum of a is >= the corresponding partial sum of b."""
    sa = sb = 0
    for i in range(max(len(a), len(b))):
        sa += a[i] if i < len(a) else 0
        sb += b[i] if i < len(b) else 0
        if sa < sb:
            return False
    return True


def conjugate(a: Partition) -> Partition:
    return tuple(sum(1 for x in a if x > j) for j in range(a[0])) if a else ()


def order_geq(n: int, normalized: bool) -> Set[Pair]:
    """All strict pairs (a, b), a != b, with a >= b in the given order (given MN_n)."""
    parts = partitions(n)
    K = {a: [kostka(a, mu) for mu in parts] for a in parts}
    f = {a: dimension(a) for a in parts}
    out = set()
    for a in parts:
        for b in parts:
            if a == b:
                continue
            al, be = (f[b], f[a]) if normalized else (1, 1)
            if all(al * x >= be * y for x, y in zip(K[a], K[b])):
                out.add((a, b))
    return out


def covers(parts: Sequence[Partition], geq: Set[Pair]) -> List[Pair]:
    return sorted((a, b) for (a, b) in geq
                  if not any((a, m) in geq and (m, b) in geq for m in parts))


def _bounds(parts, geq, x, y, upper: bool):
    le = (lambda u, v: u == v or (v, u) in geq)          # u <= v
    if upper:
        return [z for z in parts if le(x, z) and le(y, z)]
    return [z for z in parts if le(z, x) and le(z, y)]


def lattice_obstruction(parts: Sequence[Partition], geq: Set[Pair]) -> Optional[Dict]:
    """None if the poset is a lattice, else the first pair with no join or no meet."""
    le = (lambda u, v: u == v or (v, u) in geq)
    for i, x in enumerate(parts):
        for y in parts[i + 1:]:
            for upper, word in ((True, "join"), (False, "meet")):
                B = _bounds(parts, geq, x, y, upper)
                best = [z for z in B if all((le(z, w) if upper else le(w, z)) for w in B)]
                if not best:
                    return {"pair": (x, y), "missing": word,
                            "bounds": B, "minimal_bounds": [z for z in B if not any(
                                w != z and (le(w, z) if upper else le(z, w)) for w in B)]}
    return None


def height(parts: Sequence[Partition], geq: Set[Pair]) -> Tuple[int, Dict[Partition, int]]:
    """Number of elements of a longest chain, and rank[a] = length of the longest chain below a."""
    order = sorted(parts)                    # lex increasing: a linear extension (see docstring)
    rank: Dict[Partition, int] = {}
    for a in order:
        rank[a] = 1 + max((rank[b] for b in rank if (a, b) in geq), default=0)
    return max(rank.values()), rank


def width(parts: Sequence[Partition], geq: Set[Pair]) -> int:
    """Largest antichain = p(n) - maximum matching in the comparability bipartite graph (Dilworth)."""
    below = {a: [b for b in parts if (a, b) in geq] for a in parts}
    match: Dict[Partition, Partition] = {}

    def augment(a, seen):
        for b in below[a]:
            if b in seen:
                continue
            seen.add(b)
            if b not in match or augment(match[b], seen):
                match[b] = a
                return True
        return False

    m = sum(augment(a, set()) for a in parts)
    return len(parts) - m


def _chain_report(chain: Sequence[Partition], geq: Set[Pair]) -> Dict:
    steps = [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]
    bad = [(a, b) for a, b in steps if (a, b) not in geq]
    return {"length": len(chain), "monotone": not bad, "failing_steps": bad}


def structure(n: int, normalized: bool) -> Dict:
    parts = partitions(n)
    geq = order_geq(n, normalized)
    dom = {(a, b) for a in parts for b in parts if a != b and dominates(a, b)}
    assert geq <= dom, "order not contained in dominance: contradicts the proof in the docstring"
    cov = covers(parts, geq)
    h, rank = height(parts, geq)
    hooks = [(k,) + (1,) * (n - k) for k in range(n, 0, -1)]
    two_row = [(n - k, k) if k else (n,) for k in range(0, n // 2 + 1)]
    two_col = [(2,) * k + (1,) * (n - 2 * k) for k in range(n // 2, -1, -1)]
    conj_anti = all((conjugate(b), conjugate(a)) in geq for (a, b) in geq)
    return {
        "n": n, "normalized": normalized, "elements": len(parts),
        "comparable_pairs": len(geq), "dominance_pairs": len(dom),
        "incomparable_pairs": len(parts) * (len(parts) - 1) // 2 - len(geq),
        "equals_dominance": geq == dom,
        "dominance_pairs_dropped": sorted(dom - geq),
        "covers": cov,
        "dominance_covers_kept": sum(1 for p in covers(parts, dom) if p in geq),
        "dominance_covers": len(covers(parts, dom)),
        "maximal": [a for a in parts if not any((b, a) in geq for b in parts)],
        "minimal": [a for a in parts if not any((a, b) in geq for b in parts)],
        "lattice_obstruction": lattice_obstruction(parts, geq),
        "height": h, "rank": rank, "width": width(parts, geq),
        "chains": {"hooks": _chain_report(hooks, geq), "two_row": _chain_report(two_row, geq),
                   "two_column": _chain_report(two_col, geq)},
        "conjugation_anti_automorphism": conj_anti,
    }


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def compact(a: Partition) -> str:
    """Exponent notation for TikZ labels: (3,3,1,1) -> 3^2 1^2 (math mode)."""
    out, i = [], 0
    while i < len(a):
        j = i
        while j < len(a) and a[j] == a[i]:
            j += 1
        out.append(f"{a[i]}" + (f"^{{{j - i}}}" if j - i > 1 else ""))
        i = j
    return "\\,".join(out)


def hasse_tikz(n: int, normalized: bool, xstep: float = 1.25, ystep: float = 1.1) -> str:
    """A tikzpicture of the Hasse diagram; element a sits at row rank[a] (longest chain below)."""
    parts = partitions(n)
    geq = order_geq(n, normalized)
    _, rank = height(parts, geq)
    rows: Dict[int, List[Partition]] = {}
    for a in sorted(parts, reverse=True):
        rows.setdefault(rank[a], []).append(a)
    pos = {}
    for r, row in rows.items():
        for i, a in enumerate(row):
            pos[a] = ((i - (len(row) - 1) / 2) * xstep, (r - 1) * ystep)
    name = {a: "p" + "_".join(map(str, a)) for a in parts}
    lines = ["\\begin{tikzpicture}[every node/.style={font=\\scriptsize,inner sep=1.5pt}]"]
    for a in parts:
        x, y = pos[a]
        lines.append(f"  \\node ({name[a]}) at ({x:.2f},{y:.2f}) {{${compact(a)}$}};")
    for a, b in covers(parts, geq):
        lines.append(f"  \\draw[gray] ({name[a]}) -- ({name[b]});")
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)
