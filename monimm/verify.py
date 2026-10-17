"""One command for every check of paper/haiman_n10.tex (Haiman's q-conjecture for n <= 10).

    .venv/bin/python -m monimm.verify --list               # the checks, and which code runs each
    .venv/bin/python -m monimm.verify                      # tier "quick" (default)
    .venv/bin/python -m monimm.verify --tier local --json report.json
    .venv/bin/python -m monimm.verify checksums [--write]  # results/SHA256SUMS

What it does
------------
The section "Verification and results" of the paper lists thirteen checks of the computation
behind its theorem (label thm:n10: c_{lambda,w}(q) in N[q], symmetric and unimodal, for every
lambda |- 10 and w in S_10), and the appendix "Code and data" lists the code and the result files.
Section and theorem numbers are not hard-coded here, since they move between drafts.  This module is a thin layer over that code:
it runs the existing drivers unchanged, as subprocesses from the repository root, reads the
JSON summaries they write, and turns each field into a pass/fail line numbered like the paper.
It contains no mathematics of its own, so it cannot change any number in the paper.

There are two kinds of line.

* "recomputed": the driver was run now, here.  Its fresh summary is also compared field by
  field with the stored summary in results/ (all fields present in both, except timings,
  memory and paths), and the fresh coefficient tables are compared with the stored ones where
  a table is stored (results/big_n8.npz, big_n9.npz, kl_graded_n8.npz).
* "stored": a summary in results/ written by an earlier run, usually one that needs a large
  machine (n = 10: 192 cores and up to 125 GB).  The line checks the flags and counts recorded
  in the file, that the files agree with each other (the same 1,252,173,030 extremal pairs in
  every n = 10 file, ...) and with the numbers printed in the paper.  A stored line proves only
  that the recorded run passed; recomputing it needs the runbooks in cloud/.

Tiers
-----
stored   only the stored lines and the checksums (seconds, no computation).
quick    stored + recompute n <= 8 (KL tables, graded run, exact inversion), the induced
         modules of check 12 at n = 6, the worked examples, the isolation certificates and the
         test suite.  About 4 minutes on 4 cores, under 4 GB.
local    quick + recompute n = 9 (ungraded, graded, exact inversion, Coxeter3) and Sage at
         n = 6.  About an hour on 4 cores with 15 GB (graded n = 9 and Coxeter3 n = 9 take about
         24 minutes each); reproduces results/ext_n9.json and kl_graded_n9.json.

n = 10 is never recomputed here: see cloud/run_n10.sh, run_kl_graded_n10*.sh and
run_coxeter3_n10.sh, which the paper's reproduction table lists.

Optional software
-----------------
Check 2 compares with du Cloux's Coxeter3.  If the binary is found ($COXETER3, default
~/src/coxeter3/coxeter, as in cross_check_coxeter3.py) the comparison is rerun on all of S_n
(n = 8 in quick, n = 9 in local); otherwise that line is "skipped" and only the stored
comparisons (results/coxeter3_check_n{8,9,10}.json) are checked.  The SageMath part of check 9
is cross_check_sage.py: if Sage is found ($SAGE_PYTHON, a Python with SageMath importable, else
``sage -python`` when ``sage`` is on the PATH) every P_{x,w} of S_n is recomputed by Sage for
n = 2..5 (all pairs; about 35 s) and, in local, the extremal pairs of S_6 (several minutes);
otherwise that line is "skipped" and only the stored results/sage_check.json is checked.  The
reference-implementation part of check 9 is in the test suite
(test_extremal_engine_polynomials_match_reference and the tests around it).

Output
------
A table on stdout and, with --json, a report holding every line together with the environment:
Python and package versions, platform, cores, memory and the git commit.  The exit status is 0
iff no line failed (skipped lines do not count as failures).

Checksums
---------
``checksums --write`` writes results/SHA256SUMS (the format of ``sha256sum``, so
``sha256sum -c results/SHA256SUMS`` also works) for every file under results/ that git tracks,
plus results/ext_n10.npz (the full n = 10 table, not in git) when it is present.  ``checksums``
and every tier verify the list; a file that is listed but absent is reported as skipped if git
does not track it (ext_n10.npz on a fresh checkout) and as a failure otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SUMS = RESULTS / "SHA256SUMS"
# Large data files kept outside git but archived with the release.
UNTRACKED_DATA = ["results/ext_n10.npz"]
COXETER3 = os.environ.get("COXETER3", os.path.expanduser("~/src/coxeter3/coxeter"))


def sage_command():
    """The interpreter for the Sage half of cross_check_sage.py, or None."""
    if os.environ.get("SAGE_PYTHON"):
        return [os.environ["SAGE_PYTHON"]]
    if shutil.which("sage"):
        return ["sage", "-python"]
    return None

# The checks of "Verification and results", in the paper's words (shortened), and the code behind each line.
CHECKS = {
    "1": ("Sanity of the polynomials", "run_kl_extremal.py (inside monimm/kl_extremal.py)"),
    "2": ("Independent software (Coxeter3)", "cross_check_coxeter3.py"),
    "3": ("J-values c_{lambda,w0(alpha)} = alpha! delta", "run_kl_extremal.py"),
    "4": ("Symmetries w -> w^-1, w0 w w0", "run_kl_extremal.py"),
    "5": ("Kazhdan-Lusztig inversion (exact over Z)", "run_kl_extremal.py, check_inversion_exact.py"),
    "6": ("Two columns vanish on 321-containing w", "run_kl_extremal.py"),
    "7": ("Codominant w: c = Imm_lambda(A_w)", "run_kl_extremal.py"),
    "8": ("Coproduct rule", "run_kl_extremal.py"),
    "9": ("Smaller n: full-interval engine, reference, Sage", "run_kl_extremal.py --compare, tests/, cross_check_sage.py"),
    "10": ("Graded output at q = 1", "run_kl_graded.py"),
    "11": ("Graded symmetries and stability", "run_kl_graded.py"),
    "12": ("Traces from the Hecke relations", "check_graded_induced.py, tests/test_kl_graded.py"),
    "13": ("Chromatic quasisymmetric functions", "run_kl_graded.py --codominant (check_codominant_chromatic.py)"),
    "Thm": ("Main theorem: N[q], symmetric, unimodal", "run_kl_extremal.py, run_kl_graded.py"),
    "int64": ("No 64-bit overflow (a priori bounds)", "run_kl_graded.py, check_inversion_exact.py"),
    "paper": ("Numbers printed in the paper", "results/*.json"),
    "repro": ("Fresh run reproduces the stored result", "results/*.json, *.npz"),
    "Ex": ("Worked examples and figures", "paper/haiman_n10_examples.py"),
    "Iso": ("Isolation certificates, n <= 5", "check_isolation_certificates.py --verify"),
    "tests": ("Test suite", "python -m pytest -q"),
    "sha": ("Checksums of the result files", "results/SHA256SUMS"),
}

# Numbers printed in "Verification and results" for n = 10.
PAPER_N10 = {
    "extremal_pairs": 1_252_173_030,
    "distinct_polys": 981_174,
    "max_poly_degree": 12,
    "max_coefficient": 35_419_702,
    "codominant": 16_796,
    "compositions": 512,
    "coproduct_checked": 33_577_950,
    "inversion_samples": 2000,
    "inversion_log2_bound_below": 41,
}

# Summary fields that differ between runs of the same computation.
VOLATILE = {"seconds", "total_s", "timings", "timings_s", "timings_identity", "peak_rss_gb",
            "coeffs", "workers", "batch", "range", "coproduct_run", "engine"}


@dataclass
class Line:
    check: str
    n: int | None
    source: str      # "recomputed" or "stored"
    status: str      # "pass", "fail" or "skipped"
    detail: str
    files: list[str] = field(default_factory=list)


def catalan(n):
    return math.comb(2 * n, n) // (n + 1)


class Verifier:
    def __init__(self, workdir: Path, verbose=True):
        self.work = workdir
        self.lines: list[Line] = []
        self.verbose = verbose
        self.t0 = time.time()

    # -- bookkeeping -------------------------------------------------------------------------
    def add(self, check, n, source, ok, detail, files=(), skipped=False):
        status = "skipped" if skipped else ("pass" if ok else "fail")
        self.lines.append(Line(check, n, source, status, detail, list(files)))
        if self.verbose:
            print(f"  [{status:7}] {check:>5}  n={n if n is not None else '-':<2} {source:10} {detail}", flush=True)

    def say(self, msg):
        if self.verbose:
            print(f"== [{time.time() - self.t0:6.0f} s] {msg}", flush=True)

    def script(self, name, *args, log):
        """Run a driver from the repository root with this interpreter; True iff it exits 0."""
        cmd = [sys.executable, "-u", name, *map(str, args)]
        self.say(" ".join(cmd[2:]))
        with open(self.work / log, "w") as f:
            r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            self.say(f"exit status {r.returncode}; log: {self.work / log}")
        return r.returncode == 0

    @staticmethod
    def load(path):
        with open(path) as f:
            return json.load(f)

    def reproduces(self, n, fresh, stored_name):
        """Every non-volatile field present in both summaries is equal."""
        path = RESULTS / stored_name
        if not path.exists():
            return
        old = self.load(path)
        keys = sorted((set(fresh) & set(old)) - VOLATILE)
        diff = [k for k in keys if fresh[k] != old[k]]
        self.add("repro", n, "recomputed", not diff,
                 f"{len(keys)} fields equal to results/{stored_name}" if not diff else f"differs from results/{stored_name} in {diff}",
                 [f"results/{stored_name}"])

    def same_table(self, n, fresh_npz, stored_name):
        path = RESULTS / stored_name
        if not path.exists():
            return
        a, b = np.load(fresh_npz), np.load(path)
        A, B = a["C"], b["C"]
        if A.ndim == 3:   # graded tables: pad the degree axis
            D = max(A.shape[2], B.shape[2])
            A = np.pad(A, ((0, 0), (0, 0), (0, D - A.shape[2])))
            B = np.pad(B, ((0, 0), (0, 0), (0, D - B.shape[2])))
        ok = A.shape == B.shape and np.array_equal(A, B)
        self.add("repro", n, "recomputed", ok, f"coefficient table {'equal to' if ok else 'differs from'} results/{stored_name}",
                 [f"results/{stored_name}"])

    # -- the ungraded checks 1, 3-9 ----------------------------------------------------------
    def ext_lines(self, s, n, source, files):
        sanity = s["sanity"]
        self.add("1", n, source, all(sanity.values()), f"P(0) = 1 and coefficients >= 0 ({s['distinct_polys']} distinct P)", files)
        jt = s["j_test_values"]
        self.add("3", n, source, jt["mismatches"] == 0 and jt["compositions"] == 2 ** (n - 1),
                 f"{jt['compositions']} compositions, {jt['mismatches']} mismatches", files)
        self.add("4", n, source, s["symmetry_inverse"] and s["symmetry_w0_conjugation"],
                 f"w^-1: {s['symmetry_inverse']}, w0 w w0: {s['symmetry_w0_conjugation']}", files)
        si = s["sampled_identity"]
        self.add("5", n, source, si["pass"], f"sampled identity mod 2^64, {si['samples']} u", files)
        self.add("6", n, source, s["two_column_vanish_on_321"], "lambda_1 <= 2 vanishes on 321-containing w", files)
        self.add("7", n, source, s["codominant_hessenberg_check"] and s["codominant_count"] == catalan(n),
                 f"{s['codominant_count']} codominant w", files)
        if "coproduct" in s:
            cp = s["coproduct"]
            self.add("8", n, source, cp["mismatches"] == 0, f"{cp['checked']} entries, {cp['mismatches']} mismatches", files)
        if "equals_reference" in s:
            self.add("9", n, source, s["equals_reference"], "every c_{lambda,w} equals the full-interval engine (big_n%d.npz)" % n, files)
        neg = sum(v["negative"] for v in s["shapes"].values())
        self.add("Thm", n, source, s["all_nonnegative"] and neg == 0, f"q = 1: all c_(lambda,w) >= 0 ({len(s['shapes'])} shapes)", files)

    def inv_lines(self, s, n, source, files):
        ok = (s["mismatches_mod_2^64"] == 0 and s["mismatches_mod_p"] == 0 and s["no_wraparound"] and s["exact"])
        self.add("5", n, source, ok, f"exact over Z: {s['samples']} u x {s['lambdas']} lambda, also mod p = {s['p']}", files)
        self.add("int64", n, source, s["no_wraparound"] and s["log2_max_bound"] < 62,
                 f"inversion partial sums < 2^{s['log2_max_bound']:.2f}", files)

    # -- the graded checks 10, 11, 13 --------------------------------------------------------
    def graded_lines(self, s, n, source, files, codominant=None, only_chromatic=False):
        cd = codominant or s.get("codominant_chromatic_check")
        if only_chromatic:
            self.add("13", n, source, cd["mismatching_w"] == 0 and cd["codominant_w"] == catalan(n),
                     f"{cd['codominant_w']} codominant w x {cd['shapes']} shapes, {cd['mismatching_w']} mismatches", files)
            return
        self.add("10", n, source, s["check_q1_matches_kl_extremal"], "c(1) equals the ungraded coefficients", files)
        ok = s["check_inverse_symmetry"] and s["check_w0_conjugation_symmetry"] and s["check_stability_vs_n_minus_1"]
        self.add("11", n, source, ok, f"w^-1, w0 w w0 and stability against S_{n - 1}", files)
        if "check_no_int64_overflow" in s:
            self.add("int64", n, source, s["check_no_int64_overflow"] and s["accumulation_bound"] < 2 ** 63,
                     f"graded accumulation bound {s['accumulation_bound']:.3e} < 2^63", files)
        h = s["hooks"]
        ok = all(s[k] == 0 and h[k] == 0 for k in ("negative_coefficients", "asymmetric", "non_unimodal"))
        self.add("Thm", n, source, ok, f"graded: {s['negative_coefficients']} negative, {s['asymmetric']} asymmetric, "
                 f"{s['non_unimodal']} non-unimodal", files)
        if cd:
            self.add("13", n, source, cd["mismatching_w"] == 0 and cd["codominant_w"] == catalan(n),
                     f"{cd['codominant_w']} codominant w x {cd['shapes']} shapes, {cd['mismatching_w']} mismatches", files)

    def coxeter3_lines(self, s, n, source, files, pairs=None):
        ok = s["mismatching_ys"] == 0 and s["ys_checked"] == math.factorial(n) and s.get("mode") == "all"
        if pairs is not None:
            ok &= s["extremal_pairs_checked"] == pairs
        self.add("2", n, source, ok, f"{s['ys_checked']} y, {s['extremal_pairs_checked']} extremal pairs, "
                 f"{s['mismatching_ys']} mismatching y", files)

    # -- stored results ----------------------------------------------------------------------
    def stored(self):
        self.say("stored results")
        R = lambda name: RESULTS / name
        f = lambda name: [f"results/{name}"]

        def need(name):
            if R(name).exists():
                return self.load(R(name))
            self.add("repro", None, "stored", False, f"results/{name} is missing", f(name))
            return None

        pairs = {}
        for n in (9, 10):
            s = need(f"ext_n{n}.json")
            if s:
                pairs[n] = s["extremal_pairs"]
                self.ext_lines(s, n, "stored", f(f"ext_n{n}.json"))
        for n in (9, 10):
            s = need(f"inversion_exact_n{n}.json")
            if s:
                self.inv_lines(s, n, "stored", f(f"inversion_exact_n{n}.json"))
        for n in (8, 9, 10):
            s = need(f"coxeter3_check_n{n}.json")
            if s:
                self.coxeter3_lines(s, n, "stored", f(f"coxeter3_check_n{n}.json"), pairs.get(n))
        for n in (8, 9):
            s, c = need(f"kl_graded_n{n}.json"), need(f"codominant_chromatic_n{n}.json")
            if s:
                self.graded_lines(s, n, "stored", f(f"kl_graded_n{n}.json") + f(f"codominant_chromatic_n{n}.json"), c)
        g10 = need("kl_graded_n10.json")
        if g10:
            self.graded_lines(g10, 10, "stored", f("kl_graded_n10.json"))
        c10 = need("kl_graded_n10_codominant.json")
        if c10:
            self.graded_lines(c10, 10, "stored", f("kl_graded_n10_codominant.json"), only_chromatic=True)
        s = need("sage_check.json")
        if s:
            for r in s["runs"]:
                self.add("9", r["n"], "stored", r["mismatches"] == 0 and r["pairs_checked"] > 0,
                         f"SageMath {r['sage_version']}: {r['pairs_checked']} pairs ({r['mode']}), {r['mismatches']} mismatches",
                         f("sage_check.json"))
        s = need("graded_induced_n6.json")
        if s:
            self.add("12", 6, "stored", s["mismatches"] == 0 and s["w"] == 720 and s["q"] == [2, 3],
                     f"q = 2, 3 mod {s['primes']}: {s['mismatches']} mismatches", f("graded_induced_n6.json"))

        # The n = 10 files describe one table: they must agree with each other and with the paper.
        e10 = self.load(R("ext_n10.json")) if R("ext_n10.json").exists() else None
        i10 = self.load(R("inversion_exact_n10.json")) if R("inversion_exact_n10.json").exists() else None
        x10 = self.load(R("coxeter3_check_n10.json")) if R("coxeter3_check_n10.json").exists() else None
        if e10 and g10 and c10 and i10 and x10:
            P = PAPER_N10
            got = {
                "extremal_pairs": {e10["extremal_pairs"], g10["pairs_extremal"], c10["pairs_extremal"], x10["extremal_pairs_checked"]},
                "distinct_polys": {e10["distinct_polys"], g10["distinct_P"], c10["distinct_P"]},
                "max_poly_degree": {e10["max_poly_degree"]},
                "max_coefficient": {g10["max_coefficient"], c10["max_coefficient"]},
                "codominant": {e10["codominant_count"], c10["codominant_chromatic_check"]["codominant_w"]},
                "compositions": {e10["j_test_values"]["compositions"]},
                "coproduct_checked": {e10["coproduct"]["checked"]},
                "inversion_samples": {i10["samples"], e10["sampled_identity"]["samples"]},
            }
            bad = [k for k, v in got.items() if v != {P[k]}]
            if not i10["log2_max_bound"] < P["inversion_log2_bound_below"]:
                bad.append("inversion_log2_bound_below")
            self.add("paper", 10, "stored", not bad,
                     "n = 10 files agree with each other and with the paper" if not bad else f"disagree on {bad}",
                     ["results/ext_n10.json", "results/kl_graded_n10.json", "results/kl_graded_n10_codominant.json",
                      "results/inversion_exact_n10.json", "results/coxeter3_check_n10.json"])
        self.checksums()

    # -- recomputation -----------------------------------------------------------------------
    def recompute_ungraded(self, n):
        out = self.work / f"ext_n{n}"
        ref = RESULTS / f"big_n{n}.npz"
        args = ["--n", n, "--out", out] + (["--compare", ref] if ref.exists() else [])
        ok = self.script("run_kl_extremal.py", *args, log=f"ext_n{n}.log")
        if not Path(f"{out}.json").exists():
            self.add("repro", n, "recomputed", False, f"run_kl_extremal.py --n {n} wrote no summary ({self.work}/ext_n{n}.log)")
            return None
        s = self.load(f"{out}.json")
        self.ext_lines(s, n, "recomputed", [])
        if not ok:
            self.add("repro", n, "recomputed", False, "run_kl_extremal.py reported SOME CHECK FAILED")
        self.reproduces(n, s, f"ext_n{n}.json")
        inv = self.work / f"inversion_n{n}.json"
        self.script("check_inversion_exact.py", "--n", n, "--coeffs", f"{out}.npz", "--out", inv, log=f"inversion_n{n}.log")
        if inv.exists():
            si = self.load(inv)
            self.inv_lines(si, n, "recomputed", [])
            self.reproduces(n, si, f"inversion_exact_n{n}.json")
        else:
            self.add("5", n, "recomputed", False, f"check_inversion_exact.py wrote no summary ({self.work}/inversion_n{n}.log)")
        return s

    def recompute_graded(self, n, table=False):
        js = self.work / f"kl_graded_n{n}.json"
        args = ["--n", n, "--codominant", "--json", js] + (["--npz", self.work / f"kl_graded_n{n}.npz"] if table else [])
        self.script("run_kl_graded.py", *args, log=f"kl_graded_n{n}.log")
        if not js.exists():
            self.add("10", n, "recomputed", False, f"run_kl_graded.py --n {n} wrote no summary ({self.work}/kl_graded_n{n}.log)")
            return
        s = self.load(js)
        self.graded_lines(s, n, "recomputed", [])
        self.reproduces(n, {k: v for k, v in s.items() if k != "codominant_chromatic_check"}, f"kl_graded_n{n}.json")
        self.reproduces(n, s["codominant_chromatic_check"], f"codominant_chromatic_n{n}.json")
        if table:
            self.same_table(n, self.work / f"kl_graded_n{n}.npz", f"kl_graded_n{n}.npz")

    def recompute_induced(self, n=6):
        js = self.work / f"graded_induced_n{n}.json"
        ok = self.script("check_graded_induced.py", "--n", n, "--json", js, log=f"graded_induced_n{n}.log")
        if js.exists():
            s = self.load(js)
            self.add("12", n, "recomputed", ok and s["mismatches"] == 0,
                     f"{s['w']} w x {s['shapes']} shapes at q = {s['q']}, {s['mismatches']} mismatches")
            self.reproduces(n, s, f"graded_induced_n{n}.json")
        else:
            self.add("12", n, "recomputed", False, f"check_graded_induced.py wrote no summary ({self.work}/graded_induced_n{n}.log)")

    def recompute_coxeter3(self, n, pairs):
        if not os.access(COXETER3, os.X_OK):
            self.add("2", n, "recomputed", True, f"Coxeter3 not found at {COXETER3} (set $COXETER3); stored comparison only",
                     skipped=True)
            return
        js = self.work / f"coxeter3_n{n}.json"
        self.script("cross_check_coxeter3.py", "--n", n, "--all", "--workers", os.cpu_count() or 1, "--json", js,
                    log=f"coxeter3_n{n}.log")
        if js.exists():
            self.coxeter3_lines(self.load(js), n, "recomputed", [], pairs)
        else:
            self.add("2", n, "recomputed", False, f"cross_check_coxeter3.py wrote no summary ({self.work}/coxeter3_n{n}.log)")

    def small(self):
        """The examples, the isolation certificates and the test suite."""
        ok = self.script("paper/haiman_n10_examples.py", log="examples.log")
        self.add("Ex", None, "recomputed", ok, "every number in the worked examples and small figures")
        for n in (4, 5):
            name = f"tnn_isolation_certificates_n{n}.json"
            ok = self.script("check_isolation_certificates.py", "--verify", RESULTS / name, log=f"isolation_n{n}.log")
            self.add("Iso", n, "recomputed", ok, f"certificates re-verified in integer arithmetic", [f"results/{name}"])
        ok = self.script("-m", "pytest", "-q", "tests", log="pytest.log")
        tail = (self.work / "pytest.log").read_text().strip().splitlines()[-1:] or [""]
        self.add("tests", None, "recomputed", ok, tail[0].strip("= "))

    def recompute_sage(self, ns):
        cmd = sage_command()
        if cmd is None:
            self.add("9", None, "recomputed", True, "SageMath not found (set $SAGE_PYTHON); stored comparison only",
                     skipped=True)
            return
        for n in ns:
            pairs, js = self.work / f"sage_pairs_n{n}.json", self.work / f"sage_n{n}.json"
            self.script("cross_check_sage.py", "--export", n, "--out", pairs, log=f"sage_export_n{n}.log")
            self.say(f"{' '.join(cmd)} cross_check_sage.py --compare {pairs}")
            with open(self.work / f"sage_n{n}.log", "w") as f:
                subprocess.run([*cmd, "cross_check_sage.py", "--compare", pairs, "--json", js], cwd=ROOT,
                               stdout=f, stderr=subprocess.STDOUT)
            if js.exists():
                r = self.load(js)
                self.add("9", n, "recomputed", r["mismatches"] == 0 and r["pairs_checked"] > 0,
                         f"SageMath {r['sage_version']}: {r['pairs_checked']} pairs ({r['mode']}), {r['mismatches']} mismatches")
            else:
                self.add("9", n, "recomputed", False, f"cross_check_sage.py wrote no summary ({self.work}/sage_n{n}.log)")

    # -- checksums ---------------------------------------------------------------------------
    def checksums(self):
        if not SUMS.exists():
            self.add("sha", None, "stored", False, "results/SHA256SUMS is missing (run: python -m monimm.verify checksums --write)")
            return
        tracked = set(git_tracked())
        bad, absent, n_ok = [], [], 0
        for line in SUMS.read_text().splitlines():
            digest, name = line.split(None, 1)
            name = name.lstrip("*")
            p = ROOT / name
            if not p.exists():
                (absent if name not in tracked else bad).append(name)
            elif sha256(p) != digest:
                bad.append(name)
            else:
                n_ok += 1
        self.add("sha", None, "stored", not bad, f"{n_ok} files match" + (f"; mismatched or missing: {bad}" if bad else ""))
        for name in absent:
            self.add("sha", None, "stored", True, f"{name} not present (archived separately)", [name], skipped=True)


# -- helpers -----------------------------------------------------------------------------------
def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def git_tracked():
    try:
        r = subprocess.run(["git", "ls-files", "results"], cwd=ROOT, capture_output=True, text=True, check=True)
        return r.stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return []


def write_checksums():
    names = [n for n in git_tracked() if n != "results/SHA256SUMS"]
    if not names:   # not a git checkout (e.g. an unpacked archive): every file under results/
        names = sorted(str(p.relative_to(ROOT)) for p in RESULTS.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    names += [n for n in UNTRACKED_DATA if (ROOT / n).exists() and n not in names]
    SUMS.write_text("".join(f"{sha256(ROOT / n)}  {n}\n" for n in sorted(names)))
    print(f"wrote {SUMS.relative_to(ROOT)}: {len(names)} files")


def environment():
    def ver(pkg):
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                    capture_output=True, text=True).stdout.strip())
    except OSError:
        commit, dirty = None, None
    try:
        mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError, AttributeError):
        mem = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "cores": os.cpu_count(),
            "memory_gb": round(mem, 1) if mem else None, "git_commit": commit, "git_dirty": dirty,
            "packages": {p: ver(p) for p in ("numpy", "numba", "llvmlite", "scipy", "sympy", "python-flint", "pytest")},
            "coxeter3": COXETER3 if os.access(COXETER3, os.X_OK) else None}


def print_list():
    print(f"{'check':>6}  {'paper wording':46}  code")
    for k, (title, code) in CHECKS.items():
        print(f"{k:>6}  {title:46}  {code}")


def summary_table(lines):
    """The lines that did not pass (every line was printed as it came), then the counts."""
    rest = [L for L in lines if L.status != "pass"]
    if rest:
        print(f"\n{'check':>6} {'n':>3}  {'source':10} {'status':7}  detail")
    for L in rest:
        print(f"{L.check:>6} {L.n if L.n is not None else '-':>3}  {L.source:10} {L.status:7}  {L.detail}")
    counts = {s: sum(L.status == s for L in lines) for s in ("pass", "fail", "skipped")}
    print(f"\n{counts['pass']} passed, {counts['fail']} failed, {counts['skipped']} skipped")
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m monimm.verify", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="run", choices=["run", "checksums"])
    ap.add_argument("--tier", default="quick", choices=["stored", "quick", "local"])
    ap.add_argument("--write", action="store_true", help="with 'checksums': write results/SHA256SUMS")
    ap.add_argument("--json", help="write the report here")
    ap.add_argument("--workdir", help="keep the fresh outputs and logs here (default: a temporary directory)")
    ap.add_argument("--list", action="store_true", help="list the checks and the code behind each, then stop")
    a = ap.parse_args(argv)
    if a.list:
        print_list()
        return 0
    if a.command == "checksums" and a.write:
        write_checksums()
        return 0

    tmp = None
    if a.workdir:
        work = Path(a.workdir).resolve()
        work.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.mkdtemp(prefix="monimm_verify_")
        work = Path(tmp)
    V = Verifier(work)
    try:
        if a.command == "checksums":
            V.checksums()
        else:
            V.stored()
            if a.tier in ("quick", "local"):
                ns = [8] + ([9] if a.tier == "local" else [])
                pairs = {}
                for n in ns:
                    s = V.recompute_ungraded(n)
                    if s:
                        pairs[n] = s["extremal_pairs"]
                    V.recompute_graded(n, table=(n == 8))
                    V.recompute_coxeter3(n, pairs.get(n))
                V.recompute_induced(6)
                V.recompute_sage(range(2, 7 if a.tier == "local" else 6))
                V.small()
        counts = summary_table(V.lines)
        if a.json:
            report = {"tier": a.tier if a.command == "run" else "checksums", "counts": counts,
                      "seconds": round(time.time() - V.t0, 1), "environment": environment(),
                      "lines": [asdict(L) for L in V.lines]}
            Path(a.json).write_text(json.dumps(report, indent=1) + "\n")
            print(f"report: {a.json}")
        return 0 if counts["fail"] == 0 else 1
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
