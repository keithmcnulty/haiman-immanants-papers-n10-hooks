#!/usr/bin/env bash
# Check 12 at n = 10 for all 42 characters: chi^mu_q(T_v) for every v in S_10 from Hoefsmit's seminormal
# form, against the Geck-Pfeiffer traces of the graded engine (check_seminormal_traces.py).
#
# Input (uploaded next to the repo, not in git): results/kl_graded_n10_traces_modp.npz (1.2 GB), whose
# SHA-256 is checked against results/SHA256SUMS first.
# Output, all in results/ (kept for the data record):
#   seminormal_traces_n10_modp.npz      X[i, mu, v] = chi^mu_q(T_v) mod p, q = 2, 3, uint32 (1.2 GB)
#   seminormal_traces_n10.json / .log   per (mu, q): mismatching v and the SHA-256 of the column
#   seminormal_ckpt_n10/                one .npy per (mu, q), written as each finishes (resumable)
#   seminormal_traces_n8_acceptance.*   the n = 8 run against a fresh engine run, which must pass first
#
# Sized for m7a.48xlarge (192 cores; ~40 GB of matrix stacks at f = 768).
#   bash cloud/run_seminormal_n10.sh       # from the repository root
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
echo "== machine: $(nproc) cores, $(free -g | awk '/Mem:/ {print $2}') GB RAM"

want=$(grep ' results/kl_graded_n10_traces_modp.npz$' results/SHA256SUMS | cut -d' ' -f1)
have=$(sha256sum results/kl_graded_n10_traces_modp.npz | cut -d' ' -f1)
[ "$want" = "$have" ] && echo "== traces input: sha256 matches SHA256SUMS" || { echo "== traces input: sha256 MISMATCH: FAIL"; exit 1; }

.venv/bin/python -u check_seminormal_traces.py --n 8 --fresh --out-npz /dev/shm/sn8.npz \
  --json results/seminormal_traces_n8_acceptance.json > results/seminormal_traces_n8_acceptance.log 2>&1 \
  && echo "== n = 8 acceptance: PASS" || { echo "== n = 8 acceptance: FAIL"; exit 1; }

.venv/bin/python -u check_seminormal_traces.py --n 10 --cut 8 --relations-nmax 8 \
  --ckpt results/seminormal_ckpt_n10 --json results/seminormal_traces_n10.json \
  2>&1 | tee results/seminormal_traces_n10.log
python3 -c "import json,sys; d=json.load(open('results/seminormal_traces_n10.json')); sys.exit(0 if d['pass'] and d['complete'] else 1)" \
  && echo "== n = 10 seminormal check: PASS" || echo "== n = 10 seminormal check: FAIL (outputs kept)"
ls -la results/seminormal_traces_n10_modp.npz results/seminormal_traces_n10.json
echo "== done"
