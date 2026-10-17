# Code and data for two papers on monomial immanants

* **Haiman's conjecture for n ≤ 10: Kazhdan–Lusztig positivity of monomial immanants**
  (`paper/haiman_n10.tex`, PDF alongside). An exact computation proving Haiman's
  q-conjecture for every λ ⊢ n and every w ∈ S_n with n ≤ 10, together with the checks behind it.
* **Hook monomial immanants are Kazhdan–Lusztig positive** (`paper/hooks_kl.tex`, PDF alongside).
  A proof for all n. Its finite-n checks and examples are reproduced here too.

Keith McNulty, 17 October 2026.

The version cited in the papers is the release `v1.0.0`, archived as
[doi:10.5281/zenodo.23435088](https://doi.org/10.5281/zenodo.23435088).

## Quick start

Python 3.11 or later.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt && .venv/bin/pip install -e .
.venv/bin/python -m monimm.verify --list                  # the checks of the paper and the code behind each
.venv/bin/python -m monimm.verify --tier stored           # stored n <= 10 results and checksums: seconds
.venv/bin/python -m monimm.verify --tier quick            # + recompute n <= 8, examples, test suite: a few minutes
.venv/bin/python -m monimm.verify --tier local --json report.json   # + recompute n = 9: about 1 h, 15 GB
```

`monimm.verify` runs every check of the section "Verification and results" of the n ≤ 10 paper and prints
one pass/fail line per check, numbered as in the paper. It runs the drivers below unchanged and compares
their fresh output with the stored files in `results/`. The docstring of `monimm/verify.py` explains each
line. The exit status is 0 if and only if no line failed.

* **n = 10 is not recomputed** (it needs up to 125 GB and 192 cores). Its stored summaries are checked for
  their recorded flags, for agreement with each other and with the numbers printed in the paper, and
  against `results/SHA256SUMS`. The runbooks that produced them are in `cloud/`.
* **Optional cross-checks.** du Cloux's Coxeter3 (check 2) is used if `$COXETER3` or `~/src/coxeter3/coxeter`
  exists (build it from <https://github.com/tscrim/coxeter>). SageMath (part of check 9) is used if
  `$SAGE_PYTHON` points to a Python with Sage importable, or `sage` is on the PATH; for example a venv with
  `pip install passagemath-combinat passagemath-groups`. Without them those lines are reported as skipped.
* `requirements-lock.txt` pins the package versions with which the quick tier last passed.

## The full n = 10 table

`results/ext_n10.npz` (102 MB, all 42 × 10! coefficients c_{λ,w}) is too large for git. It is attached to the
`v1.0.0` release of this repository and archived on Zenodo as
[doi:10.5281/zenodo.23435128](https://doi.org/10.5281/zenodo.23435128). Put it in `results/`; then
`python -m monimm.verify checksums` checks it against `results/SHA256SUMS`, and

```sh
.venv/bin/python check_inversion_exact.py --n 10 --coeffs results/ext_n10.npz --out /tmp/inversion_n10.json   # ~10 GB
```

reruns the exact Kazhdan–Lusztig inversion check of the paper on it.

## Layout

| Path | Contents |
|---|---|
| `monimm/kl_extremal.py`, `run_kl_extremal.py` | extremal-pair KL engine (n ≤ 10), the main computation |
| `monimm/kl_graded.py`, `run_kl_graded.py` | Haiman's graded coefficients via Geck–Pfeiffer class polynomials |
| `monimm/kl_fast.py`, `monimm/kl_big.py`, `run_kl_big.py` | full-interval engines (n ≤ 8, n ≤ 9), used as independent checks |
| `monimm/kazhdan_lusztig.py`, `monimm/combinatorics.py` | reference implementation (small n), partitions, characters, monomial class functions |
| `monimm/chromatic.py`, `check_codominant_chromatic.py` | chromatic quasisymmetric functions at codominant w (check 13) |
| `check_inversion_exact.py`, `check_graded_induced.py` | exact KL inversion over Z; induced Hecke modules (check 12) |
| `cross_check_coxeter3.py`, `cross_check_sage.py` | comparisons with Coxeter3 and SageMath |
| `monimm/ordering.py`, `monimm/order_structure.py`, `immanant_ordering.py`, `order_structure.py` | the order of (normalised) immanants and its Hasse diagrams |
| `monimm/tnn_cone.py`, `check_tnn_isolation.py`, `check_isolation_certificates.py` | TNN-isolation for n ≤ 5, with exact certificates |
| `monimm/evaluate.py`, `monimm/tnn.py` | exact evaluation of immanants and totally positive test matrices (used by the tests) |
| `check_further_computations.py` | hook expansions (Conjecture 10.1 and Table 1 of the hooks paper, single blocks, inertness) and the (3,2^b,1^c) counts, n ≤ 8 |
| `notes/` | finite-n checks of the results of the hooks paper (see below) |
| `paper/` | both papers, their figures, and `*_examples.py`, which assert every number in the worked examples |
| `cloud/` | runbooks for the n = 10 runs |
| `results/` | every stored result, with `SHA256SUMS` |
| `tests/` | the test suite: every engine against brute force at small n |

## Further computations

These support the section "Further computations" of the n ≤ 10 paper and the remarks on computations in the
hooks paper. They are evidence and cross-checks, not part of either proof.

| Claim | Command | Stored result |
|---|---|---|
| Conjecture 10.1, Table 1, single blocks, inertness (n ≤ 8); 40 and 427 (3,2^b,1^c) permutations | `python check_further_computations.py --nmax 8` (~1 min) | `results/further_computations_n8.json` |
| Clean steps: coverage 84%, 68%, 49%, 32%, 19% for n = 5..9 | `python notes/clean_steps.py --n 9` | `results/clean_steps_n*.json` |
| Isolation theorem (hooks paper, Thm 9.6): lifts and Lemma 9.5 for n ≤ 10 | `python notes/pattern_lifting.py --n 9` (n = 10: ~16 GB) | `results/pattern_lifting_*.json` |
| Product theorem (hooks paper, Thm 3.1) for n ≤ 6 | `python notes/cprod.py` | |
| Its single-entry case for n ≤ 7 | `python notes/r2_entry_check.py 7` (~8 min) | |
| Skandera's hook chain (hooks paper, Cor. 5.1) | `python notes/skandera_chain.py --nmax 12 --klmax 8` (~1 min) | |
| Hook theorem lemmas (hooks paper, Section 4) | `python notes/hook_proof/<script>.py M` | |
| Order of normalised immanants | `python immanant_ordering.py`, `python order_structure.py --nmax 10` | `results/immanant_ordering_n1-12.json`, `results/order_structure_n1-10.json` |
| TNN-isolation for n = 4, 5 | `python check_tnn_isolation.py --n 5 --charts 200` (~50 min) | `results/tnn_isolation_n{4,5}.json` |

The scripts in `notes/hook_proof/` take the size m as their only argument:
`closure_lemma_check.py` (the closure identity, m ≤ 7), `closure_formulas_check.py` (rotations and H_c(e), m ≤ 9),
`bottom_lemma_check.py` (boundary signs, m ≤ 10), `recursion_check.py` (the recursion, m ≤ 7),
`bottom_diagram_check.py` (the bottom-terms diagram, m ≤ 6) and `theta_per_set_check.py` (the g_T of Skandera's θ_ℓ, n ≤ 7).

Commands are run from the repository root with the venv's Python (`.venv/bin/python`).

## Licence

The code is under the MIT licence (`LICENSE`). The data in `results/` are under the Creative Commons
Attribution 4.0 International licence ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)). The papers in
`paper/` (the `.tex` and `.pdf` files) are copyright the author; all rights reserved.
