#!/usr/bin/env python3
"""The order of irreducible character immanants on TNN_n, decided by the matrices J_mu.

Usage
-----
    python immanant_ordering.py                                   # n <= 10 in full
    python immanant_ordering.py --full 10 --nmax 12 --json results/immanant_ordering_n1-12.json

For each n and each ordered pair of partitions (a, b), decide
"Imm_{chi^a} >= Imm_{chi^b} on TNN_n", plain and normalized (divided by chi(e)).
See monimm/ordering.py for the mathematics.  Statuses:

  false                  J_mu is an explicit counterexample (unconditional)
  proved (known shapes)  difference = non-negative combination of Imm_lambda over shapes
                         with lambda TNN-positivity known (hooks, two columns, rectangles)
  proved (MN_n)          n <= 10, using MN_n from the KL computation
  proved (external)      Skandera's hook chain (normalized order only)
  proved (transitivity)  composed from proved relations
  conditional            holds iff Imm_lambda >= 0 for the listed open shapes

For n <= --full the cover relations (Hasse diagram) are printed: the complete
order, every pair decided.  The script also checks two things for every n:

* the all-n theorem in monimm/ordering.py.  When a_1 <= 2 or a = (3,1^{n-3}),
  "a >= b" is never conditional;
* Skandera's hook chain is never contradicted by a J_mu counterexample.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from monimm.combinatorics import partitions
from monimm.ordering import (MN_PROVED_NMAX, compare, cover_relations, dimension, hook,
                             proved_order)


def pstr(p) -> str:
    return "".join(map(str, p)) if max(p) < 10 else ",".join(map(str, p))


def decided_region(a) -> bool:
    """a_1 <= 2 or a = (3, 1^{n-3}): every lambda dominated by a is a known shape."""
    return a[0] <= 2 or (a[0] == 3 and (len(a) == 1 or a[1] == 1))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nmax", type=int, default=10, help="largest n (default 10)")
    parser.add_argument("--full", type=int, default=MN_PROVED_NMAX,
                        help=f"print Hasse diagrams for n <= FULL (default {MN_PROVED_NMAX})")
    parser.add_argument("--json", help="write all relations to this JSON file")
    args = parser.parse_args(argv)

    out, ok = [], True
    for n in range(1, args.nmax + 1):
        entry = {"n": n, "dimensions": {pstr(a): dimension(a) for a in partitions(n)}}
        for normalized in (False, True):
            name = "normalized" if normalized else "plain"
            rel = proved_order(n, normalized)
            counts = Counter(v["status"] for v in rel.values())

            # Checks: the all-n theorem and Skandera's chain.
            region_bad = [(a, b) for (a, b), v in rel.items()
                          if decided_region(a) and v["status"] == "conditional"]
            chain_bad = [k for k in range(2, n + 1)
                         if normalized and compare(hook(k, n), hook(k - 1, n), True)["status"] == "false"]
            ok &= not region_bad and not chain_bad

            covers = cover_relations(n, normalized)
            entry[name] = {
                "counts": dict(counts),
                "cover_relations": [[pstr(a), pstr(b), rel[(a, b)]["status"]] for a, b in covers],
                "relations": [{"a": pstr(a), "b": pstr(b), **{k: (pstr(v) if isinstance(v, tuple)
                                                                    else [pstr(x) for x in v]
                                                                    if isinstance(v, list) else v)
                                                              for k, v in r.items()}}
                              for (a, b), r in rel.items() if r["status"] != "false"],
            }
            print(f"n = {n:2d} {name:10s} " + ", ".join(f"{k}: {c}" for k, c in sorted(counts.items()))
                  + ("" if not region_bad and not chain_bad else
                     f"   CHECK FAILED: region {region_bad[:3]}, chain {chain_bad}"))
            if n <= args.full and n >= 2:
                status_of = {(a, b): rel[(a, b)]["status"] for a, b in covers}
                undecided = [c for c, s in status_of.items() if s == "conditional"]
                print("      Hasse diagram (a > b covers): "
                      + "  ".join(f"{pstr(a)}>{pstr(b)}" for a, b in sorted(covers, reverse=True))
                      + ("" if not undecided else f"   [{len(undecided)} conditional]"))
        out.append(entry)

    print(f"all-n theorem and hook-chain checks pass for n <= {args.nmax}: {ok}")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"description": "Order of irreducible character immanants on TNN_n via the "
                                      "matrices J_mu; see monimm/ordering.py. 'false' pairs are omitted "
                                      "from 'relations' (J_mu witness: compare() reproduces it).",
                       "mn_proved_nmax": MN_PROVED_NMAX, "results": out}, handle, indent=1)
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
