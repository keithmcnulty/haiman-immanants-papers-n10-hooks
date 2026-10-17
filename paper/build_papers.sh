#!/usr/bin/env bash
# Build the two papers and give the PDFs their distribution names:
#   hooks_kl.tex   -> McNulty_Hook_KL_Positivity.pdf
#   haiman_n10.tex -> McNulty_Haiman_n10.pdf
# Uses tectonic (TECTONIC overrides the path); run from anywhere.
set -euo pipefail
cd "$(dirname "$0")"
TECTONIC=${TECTONIC:-$(command -v tectonic || echo ~/micromamba/envs/tex/bin/tectonic)}
build() {
  "$TECTONIC" "$1.tex"
  mv -f "$1.pdf" "$2"
  echo "built $2"
}
build hooks_kl McNulty_Hook_KL_Positivity.pdf
build haiman_n10 McNulty_Haiman_n10.pdf
