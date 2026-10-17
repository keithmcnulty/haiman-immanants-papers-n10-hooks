#!/usr/bin/env bash
# Haiman q-conjecture for S_10 rerun with --codominant: every codominant w checked against chromatic
# quasisymmetric functions (CHSS Cor. 7.5), an independent check of the graded layer.
#
# Requirements: Ubuntu 22.04/24.04, >= 384 GB RAM (peak about 3 x 56 GB for the class
# polynomials, the traces and the output, plus the KL table), as many cores as possible,
# ~40 GB disk.  Planned on m7a.48xlarge (192 physical cores, 768 GB).
#
#   bash cloud/run_kl_graded_n10_codominant.sh       # from the repository root
#
# The n = 8 acceptance test of cloud/run_kl_graded_n10.sh is not repeated here (see below).
# The n = 10 run is capped at 90% of RAM.  It writes results/kl_graded_n10_codominant.{json,log};
# the full table (~56 GB as int64) is not saved.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi

echo "== machine: $(nproc) cores, $(free -g | awk '/Mem:/ {print $2}') GB RAM"
mkdir -p results
# (the n = 8 acceptance test was run with this code on 2026-10-15 and passed; skipped here)
CAP=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) * 1024 * 9 / 10 ))
echo "== n = 10 (address-space cap $((CAP / 1024 / 1024 / 1024)) GB)"
/usr/bin/time -v prlimit --as="$CAP" .venv/bin/python -u run_kl_graded.py --n 10 --codominant \
  --json results/kl_graded_n10_codominant.json 2>&1 | tee results/kl_graded_n10_codominant.log
echo "== done: results/kl_graded_n10_codominant.json, results/kl_graded_n10_codominant.log"
