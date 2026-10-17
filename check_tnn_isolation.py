#!/usr/bin/env python3
"""TNN cone versus KL cone on the full span of permutation monomials (see monimm/tnn_cone.py).

For each n: which KL immanants Imm_w are TNN-isolated (every f >= 0 on TNN_n then has a_w >= 0),
whether the non-isolated w are exactly the singular ones (containing 3412 or 4231), and a search
over factorisation charts for exact witnesses, i.e. f >= 0 on TNN_n with a negative KL coefficient.

Usage
-----
    python check_tnn_isolation.py --n 4
    python check_tnn_isolation.py --n 5 --charts 200 --json results/tnn_isolation_n5.json   # ~50 min

Every witness reported is re-checked in exact integer arithmetic (``verified``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from monimm.tnn_cone import (KLChartData, certified_witness, is_singular, isolated_set, perm_name,
                             random_charts, standard_chart, verify_witness)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--charts", type=int, default=50, help="charts to search for witnesses")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
    data = KLChartData(args.n)
    iso = isolated_set(data, seed=args.seed)
    non = sorted(w for w in data.perms if w not in iso)
    singular = sorted(w for w in data.perms if is_singular(w))
    log(f"isolated {len(iso)}/{len(data.perms)}; non-isolated == singular: {non == singular}")

    witnesses = {}
    charts = [standard_chart(args.n)] + list(random_charts(args.n, args.charts - 1, args.seed))
    for c, seq in enumerate(charts):
        if (c + 1) % 25 == 0:
            log(f"{c + 1}/{len(charts)} charts, witnesses for {len(witnesses)}/{len(non)}")
        todo = [w for w in non if perm_name(w) not in witnesses]
        if not todo:
            break
        K, _ = data.chart_coefficients(seq)      # once per chart, shared by every w
        for w in todo:
            if perm_name(w) in witnesses:
                continue
            sol = certified_witness(data, w, seq, K)
            if sol is None:
                continue
            combo = {v: round(a) for v, a in sol.items()}
            exact = all(abs(a - round(a)) < 1e-9 for a in sol.values()) and verify_witness(data, combo, seq)
            witnesses[perm_name(w)] = {"chart": c, "sequence": seq, "verified": exact,
                                       "combination": {perm_name(v): (combo[v] if exact else sol[v])
                                                       for v in sol}}
            log(f"witness for {perm_name(w)} in chart {c} (exactly verified: {exact}): "
                f"{witnesses[perm_name(w)]['combination']}")
    summary = {"n": args.n, "isolated": len(iso), "total": len(data.perms),
               "non_isolated": [perm_name(w) for w in non],
               "non_isolated_equals_singular": non == singular,
               "charts_searched": len(charts), "witnesses": witnesses,
               "seconds": round(time.time() - t0, 1)}
    log(json.dumps({k: v for k, v in summary.items() if k != "witnesses"}))
    if args.json:
        json.dump(summary, open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
