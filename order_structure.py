#!/usr/bin/env python3
"""Structure of the plain and normalized immanant orders on TNN_n (see monimm/order_structure.py).

Usage
-----
    python order_structure.py --nmax 10 --json results/order_structure_n1-10.json --tikz paper/figures

For each n <= nmax and each order, this reports:
* comparable pairs, compared with dominance order, which contains both orders (proved);
* covers, height, width, and the maximal and minimal elements;
* lattice or not, with the first pair lacking a join or meet;
* whether the hook, two-row and two-column chains are monotone;
* whether conjugation reverses the order.

With --tikz it writes one Hasse-diagram file per n and order (hasse_{plain,normalized}_n{n}.tex),
plus hasse_preview.tex, which includes them all.  The results are complete (not conditional) for
n <= MN_PROVED_NMAX = 10.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from monimm.ordering import MN_PROVED_NMAX
from monimm.order_structure import hasse_tikz, structure


def pstr(p) -> str:
    return "".join(map(str, p)) if max(p) < 10 else ",".join(map(str, p))


def jsonable(d: dict) -> dict:
    lo = d["lattice_obstruction"]
    return {
        **{k: d[k] for k in ("n", "normalized", "elements", "comparable_pairs", "dominance_pairs",
                             "incomparable_pairs", "equals_dominance", "dominance_covers",
                             "dominance_covers_kept", "height", "width",
                             "conjugation_anti_automorphism")},
        "dominance_pairs_dropped": [[pstr(a), pstr(b)] for a, b in d["dominance_pairs_dropped"]],
        "covers": [[pstr(a), pstr(b)] for a, b in d["covers"]],
        "maximal": [pstr(a) for a in d["maximal"]], "minimal": [pstr(a) for a in d["minimal"]],
        "lattice": lo is None,
        "lattice_obstruction": None if lo is None else {
            "pair": [pstr(x) for x in lo["pair"]], "missing": lo["missing"],
            "minimal_bounds": [pstr(x) for x in lo["minimal_bounds"]]},
        "rank": {pstr(a): r for a, r in d["rank"].items()},
        "chains": {k: {"monotone": v["monotone"],
                       "failing_steps": [[pstr(a), pstr(b)] for a, b in v["failing_steps"]]}
                   for k, v in d["chains"].items()},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nmax", type=int, default=MN_PROVED_NMAX)
    ap.add_argument("--json")
    ap.add_argument("--tikz", help="directory for Hasse-diagram TikZ files")
    args = ap.parse_args(argv)
    if args.nmax > MN_PROVED_NMAX:
        print(f"warning: n > {MN_PROVED_NMAX} assumes MN_n, which is not proved there")
    out = []
    for n in range(1, args.nmax + 1):
        for normalized in (False, True):
            d = structure(n, normalized)
            j = jsonable(d)
            out.append(j)
            lo = j["lattice_obstruction"]
            print(f"n={n:2d} {'normalized' if normalized else 'plain':10s} "
                  f"comparable {j['comparable_pairs']:4d}/{j['dominance_pairs']:4d} dominance, "
                  f"covers {len(j['covers']):3d}, height {j['height']:2d}, width {j['width']:2d}, "
                  f"{'lattice' if lo is None else 'no ' + lo['missing'] + ' for ' + '/'.join(lo['pair'])}; "
                  f"monotone chains: {[k for k, v in j['chains'].items() if v['monotone']]}")
            if args.tikz and n >= 2:
                os.makedirs(args.tikz, exist_ok=True)
                with open(os.path.join(args.tikz, f"hasse_{'normalized' if normalized else 'plain'}_n{n}.tex"), "w") as h:
                    h.write(hasse_tikz(n, normalized) + "\n")
    if args.tikz:
        with open(os.path.join(args.tikz, "hasse_preview.tex"), "w") as h:
            h.write("\\documentclass{article}\n\\usepackage[margin=0.5in,landscape]{geometry}\n"
                    "\\usepackage{tikz}\n\\pagestyle{empty}\n\\begin{document}\n")
            for n in range(2, args.nmax + 1):
                for kind in ("plain", "normalized"):
                    h.write(f"\\section*{{$n={n}$, {kind}}}\n\\begin{{center}}\\resizebox{{!}}{{0.8\\textheight}}"
                            f"{{\\input{{hasse_{kind}_n{n}.tex}}}}\\end{{center}}\n\\clearpage\n")
            h.write("\\end{document}\n")
    if args.json:
        with open(args.json, "w") as h:
            json.dump({"description": "Structure of the plain and normalized orders of irreducible "
                                      "character immanants on TNN_n (monimm/order_structure.py). "
                                      f"Complete for n <= {MN_PROVED_NMAX} (MN_n proved).",
                       "mn_proved_nmax": MN_PROVED_NMAX, "results": out}, h, indent=1)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
