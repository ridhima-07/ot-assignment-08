#!/bin/bash
# Reproduce every experiment. Results go to results/<track>/*.json and *.png.
# Usage: ./run_all.sh [tracks...]     e.g. ./run_all.sh A B   (default: all)
# Prerequisites: see README.md (Python packages, gcc, PostgreSQL 16 + HypoPG, QEMU + RISC-V gcc).
set -e
cd "$(dirname "$0")"
XV6=${XV6:-external/xv6-riscv}          # built by trackG/setup_xv6.sh
MORE=${MORE:-external/more-itertools}   # git clone https://github.com/more-itertools/more-itertools
TRACKS=${@:-A B C D E F G H I J}
log() { echo -e "\n===== $1 ($(date +%H:%M:%S))"; }
for t in $TRACKS; do
  case $t in
    A) log "Track A"; python3 trackA/landscapes.py >/dev/null; python3 trackA/ml_optim.py >/dev/null ;;
    B) log "Track B"; python3 trackB/discrete.py >/dev/null ;;
    C) log "Track C"; python3 trackC/quantum_opt.py --fresh >/dev/null ;;
    D) log "Track D"; python3 trackD/submodular.py >/dev/null; python3 trackD/parallel.py >/dev/null; python3 trackD/distributed.py >/dev/null ;;
    E) log "Track E"; python3 trackE/nurse.py >/dev/null ;;
    F) log "Track F"; python3 trackF/run_trackF.py >/dev/null ;;
    G) log "Track G"; [ -d "$XV6" ] || trackG/setup_xv6.sh "$XV6"
       python3 trackG/bench_sched.py "$XV6" 3 >/dev/null; python3 trackG/bench_cow.py "$XV6" 3 >/dev/null ;;
    H) log "Track H"; trackH/load.sh "${TPCH_DIR:-external/tpch}" >/dev/null; python3 trackH/index_selection.py >/dev/null ;;
    I) log "Track I"; python3 trackI/quantum_algorithms.py >/dev/null ;;
    J) log "Track J"; python3 trackJ/test_minimization.py "$MORE" >/dev/null ;;
  esac
done
log "unit tests"; python3 -m pytest -q -p no:cacheprovider tests
python3 docs/summarize.py   # regenerates docs/RESULTS_SUMMARY.md from results/*.json
