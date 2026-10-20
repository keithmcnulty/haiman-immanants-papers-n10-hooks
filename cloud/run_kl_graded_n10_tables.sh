#!/usr/bin/env bash
# Rerun of the S_10 graded computation (cloud/run_kl_graded_n10.sh), this time keeping the tables:
#   results/kl_graded_n10.npz.zst          full table C[lambda, w, k] (np.savez, then zstd)
#   results/kl_graded_n10_traces_modp.npz  phi^lambda_q(T_v) mod p at q = 2, 3, every v (uint32)
#   results/kl_graded_n10_rerun.{json,log} summary, which must equal results/kl_graded_n10.json
# These feed the n = 10 induced-module check (extension of check 12 of paper/haiman_n10.tex).
#
# Requirements as for run_kl_graded_n10.sh (m7a.48xlarge, 768 GB), plus ~70 GB free in /dev/shm.
#
#   bash cloud/run_kl_graded_n10_tables.sh       # from the repository root
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
command -v zstd >/dev/null || sudo apt-get install -y -q zstd

echo "== machine: $(nproc) cores, $(free -g | awk '/Mem:/ {print $2}') GB RAM"
mkdir -p results
strip() { grep -v '_s"\|"tables\|"kl"\|lists\|traces\|coeff\|accumulation\|overflow\|q-coeff' "$1"; }

# n = 8 acceptance: same summary and the identical full table as the stored run
.venv/bin/python -u run_kl_graded.py --n 8 --json results/kl_graded_n8_acceptance.json \
  --npz /dev/shm/kl_graded_n8.npz --raw --traces-npz /dev/shm/kl_graded_n8_traces.npz \
  > results/kl_graded_n8_acceptance.log 2>&1
if diff <(strip results/kl_graded_n8.json) <(strip results/kl_graded_n8_acceptance.json) && \
   .venv/bin/python -c "
import numpy as np, sys
a = np.load('results/kl_graded_n8.npz'); b = np.load('/dev/shm/kl_graded_n8.npz')
sys.exit(0 if a['C'].shape == b['C'].shape and (a['C'] == b['C']).all() else 1)"; then
  echo "== n = 8 acceptance: PASS"
else
  echo "== n = 8 acceptance: FAIL"; exit 1
fi

CAP=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) * 1024 * 9 / 10 ))
echo "== n = 10 (address-space cap $((CAP / 1024 / 1024 / 1024)) GB)"
/usr/bin/time -v prlimit --as="$CAP" .venv/bin/python -u run_kl_graded.py --n 10 \
  --json results/kl_graded_n10_rerun.json --npz /dev/shm/kl_graded_n10.npz --raw \
  --traces-npz results/kl_graded_n10_traces_modp.npz 2>&1 | tee results/kl_graded_n10_rerun.log

if diff <(strip results/kl_graded_n10.json) <(strip results/kl_graded_n10_rerun.json); then
  echo "== n = 10 summary equals the stored run: PASS"
else
  echo "== n = 10 summary differs from the stored run: FAIL (tables kept)"
fi
echo "== compressing the table"
zstd -T0 -3 -q /dev/shm/kl_graded_n10.npz -o results/kl_graded_n10.npz.zst
(cd /dev/shm && sha256sum kl_graded_n10.npz) > results/kl_graded_n10.npz.sha256
ls -la results/kl_graded_n10.npz.zst results/kl_graded_n10_traces_modp.npz
echo "== done: results/kl_graded_n10.npz.zst, results/kl_graded_n10_traces_modp.npz, results/kl_graded_n10_rerun.json"
