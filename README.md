# Optimize Anything — Assignment 8 (14-Day Vibe-Coding Challenge)

**Ridhima Trivedi · 241071062 · TY BTech CS, Batch D · Optimization Techniques**

All ten compulsory tracks (A–J) are implemented, run, verified and measured. Main case study (Track E): **hospital
nurse rostering**. Every component was produced with AI assistance (Claude Opus 5.5) and is disclosed in
[`docs/AI_TOOLS.md`](docs/AI_TOOLS.md); every commit is prefixed `[AI:claude-opus-5.5]`.

| Document | What it is |
|---|---|
| [`REPORT.md`](REPORT.md) | Technical report: one section per track, trade-offs, limitations, answers to the 10 final questions |
| [`docs/RESULTS_SUMMARY.md`](docs/RESULTS_SUMMARY.md) | Every results table, generated from `results/*/*.json` by `docs/summarize.py` |
| [`docs/CHAPTER6_CHECKLIST.md`](docs/CHAPTER6_CHECKLIST.md) | Every Chapter 6 method → experiment, code location, result |
| [`docs/FAILURE_LOG.md`](docs/FAILURE_LOG.md) | 15 cases where AI-generated code was wrong, and how each was caught |
| [`docs/AI_TOOLS.md`](docs/AI_TOOLS.md) | AI tools register and what the student must still do |
| [`docs/TOOL_COMPARISON.md`](docs/TOOL_COMPARISON.md) | Claude Opus 5.5 vs Claude Haiku 5.5 on the same task, graded by the same fuzzer |
| [`docs/VIVA_NOTES.md`](docs/VIVA_NOTES.md) | How each track works, likely viva questions, live-modification exercises |
| [`prompts/`](prompts) | Prompt log per track (`prompts/<A..J>/LOG.md`) and the session log |

## Repository layout

```
common/            shared helpers (seeding, timing, JSON, plotting)
trackA/  continuous optimisation   optimizers.py (from scratch), landscapes.py, ml_optim.py
trackB/  discrete + QUBO/Ising      discrete.py
trackC/  quantum optimisation       quantum_opt.py (QAOA, VQE, param-shift, SPSA, QNG, shots, noise, barren plateaus)
trackD/  submodular / parallel / distributed   submodular.py, par_bench.c (pthreads), parallel.py, distributed.py
trackE/  MAIN CASE STUDY: nurse rostering      nurse.py
trackF/  compiler                   minic.py (lexer/parser/IR/VM/C backend), passes.py, fuzz.py, run_trackF.py, programs/*.mc
trackG/  xv6                        xv6-a8.patch (MLFQ scheduler + copy-on-write fork + 4 syscalls), setup_xv6.sh,
                                    qemu_run.py, bench_sched.py, bench_cow.py, usertests_*.log
trackH/  DBMS                       schema.sql, load.sh, index_selection.py (PostgreSQL 16 + HypoPG, TPC-H SF 0.1)
trackI/  quantum algorithms + QML   quantum_algorithms.py
trackJ/  software engineering       test_minimization.py (more-itertools test suite)
tests/   fast unit tests for the claims the report relies on
results/ JSON + PNG output of every experiment
```

## Setup (Ubuntu 24.04; tested in the environment described below)

```bash
# Python 3.12+ packages
pip install numpy scipy scikit-learn matplotlib networkx pandas qiskit qiskit-aer qiskit-ibm-runtime \
            minorminer dwave-networkx psycopg2-binary pytest coverage highspy
# C toolchain, xv6 toolchain and emulator, DBMS
sudo apt install build-essential qemu-system-misc gcc-riscv64-linux-gnu postgresql-16 postgresql-16-hypopg
# External code (not committed): xv6, TPC-H dbgen, the Track J subject project
mkdir -p external && cd external
git clone https://github.com/more-itertools/more-itertools.git && (cd more-itertools && git checkout 08dca284)
git clone https://github.com/electrum/tpch-dbgen.git && (cd tpch-dbgen && make CC=gcc DATABASE=ORACLE MACHINE=LINUX WORKLOAD=TPCH)
mkdir tpch && cd tpch && cp ../tpch-dbgen/dists.dss . && ../tpch-dbgen/dbgen -s 0.1 -f && cd ../..
trackG/setup_xv6.sh external/xv6-riscv          # clones xv6 @ e93dabe, applies the patch, builds
# PostgreSQL: a superuser role for your login and a database called tpch
sudo -u postgres createuser -s $USER && createdb tpch
```

## Run

```bash
./run_all.sh            # every track (≈ 45 min on 2 vCPUs); or e.g. ./run_all.sh A E F
python3 -m pytest -q tests
python3 trackG/qemu_run.py external/xv6-riscv --smp 3 "cowbench 1 0" "schedbench 1 0" "usertests"
```

All experiments use fixed seeds; stochastic methods are repeated over 3–10 seeds and reported as mean ± std.

## Environment used for the reported numbers

2 vCPU / 7 GB Linux VM (cloud), Python 3.13, NumPy 2.5, SciPy 1.18 (HiGHS), Qiskit 2.6 + Aer 0.18, gcc 13.3,
QEMU 8.2.2 (TCG, RISC-V `virt`), PostgreSQL 16.15 + HypoPG 1.4, parallel query and JIT disabled for stable timings.
**No real quantum hardware** was used: quantum results are exact simulation, a synthetic noise model (Track C) and
the FakeManilaV2 calibration-based noise model (Track I).

## Headline results (details and caveats in the report)

| Track | Result |
|---|---|
| A | Newton/Newton-CG/natural gradient reach 1e-8 in 7 iterations vs 1,638 for GD; plain Newton converges *to the saddle*, first-order methods escape; backprop gradient check 4.5e-10 |
| B | DP = ILP = B&B on every instance; QUBO ↔ Ising energies identical on all 2⁸ assignments; penalty P < 1 gives infeasible selections |
| C | QNG reaches chemical accuracy in 4 iterations (param-shift GD 18); QAOA (ratio ≤ 0.84) does **not** beat simulated annealing on this instance |
| D | Work stealing/cyclic/dynamic ≈ 2.0× on 2 cores vs 1.1× for block split; FedAvg client drift on non-IID data; GradNorm best mean task loss |
| E | 20-nurse ward: penalty 2,458 (manual) → 448 (greedy) → 186 (annealing) → **37 (ILP, proved optimal in 1.7 s)** |
| F | 4+ passes, IR instructions −20 to −60%, 1,000 random programs × 3 pipelines with 0 output mismatches |
| G | COW fork 16× faster with a 2 MB heap, 98.6% less memory for 8 children (1.9× slower if children rewrite everything); MLFQ: interactive turnaround −41%; usertests pass |
| H | Index selection: weighted workload ≈ 6× faster with an 11 MB budget; ILP ≈ 10% better than greedy at the tightest budget; identical query results |
| I | All taught algorithms hand-written; Grover's query advantage is not a wall-clock one in simulation; VQC 93% vs RBF-SVM 97% |
| J | 769 → 266 tests at 100% line coverage, but no time saved (one test = 65% of runtime) and mutation score drops to 78.6% |

## Academic honesty

This repository was generated in one agentic AI session from the request in `prompts/general/LOG.md`
(transcript: `prompts/general/CHAT_TRANSCRIPT.md`). The tool comparison (Opus 5.5 vs Haiku 5.5) is in
`docs/TOOL_COMPARISON.md`. Two things only the student can do: commit her own edits as `[manual]`, and explain and
modify the code live at the viva (Section 8). See `docs/AI_TOOLS.md`.
