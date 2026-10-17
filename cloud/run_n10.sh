#!/usr/bin/env bash
# Kazhdan-Lusztig positivity of monomial immanants for S_10 (extremal-pair engine).
#
# Requirements: Ubuntu 22.04/24.04, >= 32 GB RAM, as many cores as possible (the run is
# compute-bound; ~13 GB peak), ~20 GB free disk.  Planned on c7a.48xlarge (192 physical cores).
#
#   bash cloud/run_n10.sh            # from the repository root
#   SKIP_ACCEPTANCE=1 EXTRA="--skip-small" bash cloud/run_n10.sh   # fastest cloud run; do
#                                    # the n = 9 test and the coproduct check elsewhere
#
# The n = 9 acceptance test runs first and must reproduce results/big_n9.npz exactly
# (ALL CHECKS PASS) before n = 10 starts.  The n = 10 run is capped at 90% of RAM.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi

echo "== machine: $(nproc) cores, $(free -g | awk '/Mem:/ {print $2}') GB RAM"
mkdir -p results
if [ -z "${SKIP_ACCEPTANCE:-}" ]; then
  .venv/bin/python -u run_kl_extremal.py --n 9 --out results/ext_n9 --compare results/big_n9.npz \
    2>&1 | grep -v '^  level' | tail -3 | tee results/ext_n9_acceptance.log
  grep -q "ALL CHECKS PASS" results/ext_n9_acceptance.log
fi

CAP=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) * 1024 * 9 / 10 ))
echo "== n = 10 (address-space cap $((CAP / 1024 / 1024 / 1024)) GB)"
/usr/bin/time -v prlimit --as="$CAP" .venv/bin/python -u run_kl_extremal.py --n 10 \
  --out results/ext_n10 ${EXTRA:-} 2>&1 | tee results/ext_n10.log
echo "== done: results/ext_n10.json, results/ext_n10.npz, results/ext_n10.log"
