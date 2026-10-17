#!/usr/bin/env python3
"""Cross-check the extremal-pair KL engine against du Cloux's Coxeter3.

Coxeter3 (F. du Cloux; Sage's maintained fork, github.com/tscrim/coxeter) is an
independent C++ implementation of Kazhdan-Lusztig polynomials.  Its ``extremals``
command prints, for an element y, the list of extremal x <= y (the x with
D_L(x) >= D_L(y) and D_R(x) >= D_R(y)) together with P_{x,y}.  That is exactly
what :class:`monimm.kl_extremal.ExtremalKL` stores for y, so we compare the two
lists entry by entry: same set of x, same polynomials.

Conventions.  Coxeter3 generator a (1-indexed) is our s_{a-1}, which swaps the
values a-1 and a.  A word a_1 ... a_k is the product s_{a_1-1} ... s_{a_k-1},
built by left multiplication from the right end.  We feed y as the reduced word
obtained by repeatedly stripping a left descent, and parse every x Coxeter3
prints back into a permutation the same way.  Since P_{x,y} = P_{x^{-1},y^{-1}} =
P_{w0 x w0, w0 y w0}, a mismatch between the two conventions could not hide
in the comparison: the check at small n compares every y, and any convention
error would show up there as different x lists.

Usage
-----
    python cross_check_coxeter3.py --n 6 --all                    # every y in S_6
    python cross_check_coxeter3.py --n 10 --samples 300 --seed 1  # random y in S_10
    python cross_check_coxeter3.py --n 10 --all --workers 190 --batch 20   # all of S_10

Coxeter3 is single-threaded and keeps everything it computes, so sessions of
about 20 y are cheapest (S_10: ~0.13 s and ~0.4 GB per session of 20).  With
``--workers`` the batches run in forked worker processes that share the engine.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time

import numpy as np

from monimm.kl_extremal import ExtremalKL

COXETER = os.environ.get("COXETER3", os.path.expanduser("~/src/coxeter3/coxeter"))

# The engine, shared with forked worker processes (read-only, copy-on-write).
_E = None
_ARGS = None


def reduced_word(p):
    """Letters a_1..a_k (Coxeter3 numbering) with p = s_{a_1-1} ... s_{a_k-1}."""
    p = list(p)
    n = len(p)
    word = []
    while True:
        pos = {v: i for i, v in enumerate(p)}
        i = next((i for i in range(n - 1) if pos[i] > pos[i + 1]), None)
        if i is None:
            return word
        word.append(i + 1)
        a, b = pos[i], pos[i + 1]           # s_i p: swap the values i and i+1
        p[a], p[b] = i + 1, i


def word_to_perm(word, n):
    p = list(range(n))
    for a in reversed(word):                # s_{a_1} (... (s_{a_k} e))
        i = a - 1
        pos = {v: j for j, v in enumerate(p)}
        p[pos[i]], p[pos[i + 1]] = i + 1, i
    return tuple(p)


def parse_poly(s):
    """'1+2q+q^3' -> [1, 2, 0, 1]."""
    coeffs = {}
    for term in s.split("+"):
        m = re.fullmatch(r"(\d*)(q(?:\^(\d+))?)?", term)
        if not m or term == "":
            raise ValueError(f"cannot parse polynomial {s!r}")
        c = int(m.group(1)) if m.group(1) else 1
        d = 0 if not m.group(2) else (int(m.group(3)) if m.group(3) else 1)
        coeffs[d] = coeffs.get(d, 0) + c
    out = [0] * (max(coeffs) + 1)
    for d, c in coeffs.items():
        out[d] = c
    return out


def run_coxeter3(n, words, timeout):
    """One Coxeter3 session; returns the raw stdout."""
    lines = ["type", "A", str(n - 1)]
    for w in words:
        lines += ["extremals", " ".join(map(str, w)) if w else "", ""]
    lines += ["qq"]
    res = subprocess.run([COXETER], input="\n".join(lines) + "\n", capture_output=True,
                         text=True, timeout=timeout, cwd=os.path.dirname(COXETER))
    return res.stdout


def parse_blocks(out, count):
    """Split the session output into the ``count`` extremals listings."""
    marker = "Name an output file (hit return for stdout):"
    parts = out.split(marker)[1:]
    if len(parts) != count:
        raise RuntimeError(f"expected {count} listings, got {len(parts)}")
    blocks = []
    for part in parts:
        body = part.split("coxeter :")[0]
        # Coxeter3 folds lines at 78 characters; a long polynomial continues on the
        # next line as "    +q^11".  Join the continuation lines first.
        body = re.sub(r"\s*\n\s+\+", "+", body)
        # the identity is printed as an empty word: " : 1"
        entries = re.findall(r"(\d*)\s*:\s*([0-9q^+]+)", body)
        blocks.append([(tuple(int(c) for c in w), parse_poly(p)) for w, p in entries])
    return blocks


def check_batch(batch):
    """Run one Coxeter3 session on the y's of ``batch`` and compare with the engine."""
    E, args = _E, _ARGS
    T, n = E.tables, E.n
    words = [reduced_word(T.perms[y]) for y in batch]
    out = run_coxeter3(n, words, args.timeout)
    pairs = mism = 0
    bad = []
    for y, block in zip(batch, parse_blocks(out, len(batch))):
        theirs = {T.rank(word_to_perm(w, n)): p for w, p in block}
        a, m = int(E.start[y]), int(E.size[y])
        ours = {}
        for t in range(a, a + m):
            row = [int(c) for c in E.polys[E.pid[t]]]
            while row and row[-1] == 0:
                row.pop()
            ours[int(E.xs[t])] = row
        pairs += len(ours)
        if ours != theirs:
            mism += 1
            only_ours = sorted(set(ours) - set(theirs))
            only_theirs = sorted(set(theirs) - set(ours))
            differ = [x for x in ours if x in theirs and ours[x] != theirs[x]]
            perm = lambda r: [int(v) for v in T.perms[r]]
            bad.append({"y": perm(y), "word": words[batch.index(y)],
                        "ours": len(ours), "coxeter3": len(theirs),
                        "only_ours": [perm(x) for x in only_ours[:3]],
                        "only_coxeter3": [perm(x) for x in only_theirs[:3]],
                        "differ": [{"x": perm(x), "ours": ours[x], "coxeter3": theirs[x]}
                                   for x in differ[:3]]})
    return len(batch), pairs, mism, bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--all", action="store_true", help="check every y in S_n")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=50, help="y's per Coxeter3 session")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--json", help="write a summary here")
    ap.add_argument("--workers", type=int, default=1, help="parallel Coxeter3 sessions")
    ap.add_argument("--range", help="lo:hi, check only this slice of the (shuffled) y list")
    args = ap.parse_args(argv)
    n = args.n
    if n > 10:
        raise SystemExit("generators are parsed as single digits; n <= 10 only")
    t0 = time.time()
    E = ExtremalKL(n, verbose=n >= 9)
    T = E.tables
    print(f"engine built in {time.time() - t0:.0f} s", flush=True)
    if args.all:
        ys = np.arange(T.N)
    else:
        rng = np.random.default_rng(args.seed)
        ys = rng.choice(T.N, size=min(args.samples, T.N), replace=False)
        ys = np.concatenate([ys, [int(T.order[-1])]])          # always include w_0
    if args.all:
        ys = np.random.default_rng(args.seed).permutation(ys)   # mix long and short y
    if args.range:
        lo, hi = (int(v) for v in args.range.split(":"))
        ys = ys[lo:hi]
    batches = [[int(y) for y in ys[b:b + args.batch]] for b in range(0, len(ys), args.batch)]
    global _E, _ARGS
    _E, _ARGS = E, args
    checked_pairs = mismatches = done = 0
    bad = []
    if args.workers > 1:
        pool = mp.get_context("fork").Pool(args.workers)
        results = pool.imap_unordered(check_batch, batches, chunksize=4)
    else:
        results = map(check_batch, batches)
    step = max(1, len(batches) // 50)
    for k, (cnt, pairs, mism, ex) in enumerate(results):
        done += cnt
        checked_pairs += pairs
        mismatches += mism
        for e in ex:                                 # report mismatches at once
            print("  MISMATCH " + json.dumps(e), flush=True)
        bad.extend(ex[:max(0, 20 - len(bad))])
        if (k + 1) % step == 0 or k + 1 == len(batches):
            print(f"  {done}/{len(ys)} y checked, {checked_pairs} pairs, "
                  f"{mismatches} mismatching y, {time.time() - t0:.0f} s", flush=True)
    if args.workers > 1:
        pool.close()
        pool.join()
    summary = {"n": n, "ys_checked": int(len(ys)), "extremal_pairs_checked": checked_pairs,
               "mismatching_ys": mismatches, "examples": bad,
               "mode": "all" if args.all else f"random {args.samples} + w0, seed {args.seed}",
               "workers": args.workers, "batch": args.batch, "range": args.range,
               "seconds": round(time.time() - t0, 1)}
    print(json.dumps(summary, indent=1))
    if args.json:
        json.dump(summary, open(args.json, "w"), indent=1)
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
