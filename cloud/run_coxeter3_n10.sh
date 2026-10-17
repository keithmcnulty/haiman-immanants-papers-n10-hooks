#!/usr/bin/env bash
# Cross-check every extremal pair of S_10 against du Cloux's Coxeter3.
#
# Requirements: Ubuntu 24.04, g++/make, ~100 GB RAM, as many cores as possible.
# Planned on c7a.48xlarge (192 cores): ~45-65 min.  Expects the Coxeter3 source
# (github.com/tscrim/coxeter) in ~/src/coxeter3.
#
#   bash cloud/run_coxeter3_n10.sh          # from the repository root
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
( cd ~/src/coxeter3 && make -j"$(nproc)" optimize=true executable > build.log 2>&1 && make check optimize=true >> build.log 2>&1 )
echo "== coxeter3 built and self-tested"

echo "== machine: $(nproc) cores, $(free -g | awk '/Mem:/ {print $2}') GB RAM"
mkdir -p results
.venv/bin/python -u cross_check_coxeter3.py --n 8 --all --workers "$(nproc)" --batch 20 | tail -3
WORKERS=$(( $(nproc) - 2 ))
/usr/bin/time -v .venv/bin/python -u cross_check_coxeter3.py --n 10 --all --workers "$WORKERS" \
  --batch 20 --json results/coxeter3_check_n10.json 2>&1 | tee results/coxeter3_check_n10.log
echo "== done: results/coxeter3_check_n10.json"
