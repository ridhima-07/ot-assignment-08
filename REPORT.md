# Optimize Anything — Technical Report

**Ridhima Trivedi · 241071062 · TY BTech CS, Batch D** · Optimization Techniques, Assignment 8
Repository: this folder · All tables: [`docs/RESULTS_SUMMARY.md`](docs/RESULTS_SUMMARY.md) (generated from `results/`) ·
AI disclosure: [`docs/AI_TOOLS.md`](docs/AI_TOOLS.md) · Failures: [`docs/FAILURE_LOG.md`](docs/FAILURE_LOG.md)

> **Reading guide.** Each track follows the Section 4 workflow: process → baseline → formulation → method choice →
> results → correctness check → trade-offs. "Measured" means timed on the test machine (2 vCPUs); "simulated" means a
> model or a quantum simulator; "theoretical" means derived, not measured. Stochastic results are mean ± std over
> 3–10 seeds.

---

## 1. Track E — Main case study: hospital nurse rostering

### 1.1 Process and evidence it needs optimisation
A ward in-charge builds a 14-day roster for 10–80 nurses over three shifts (Morning 07–14, Evening 14–21, Night
21–07). Done by hand — filling each shift from the top of the staff list — the *baseline* roster for a 20-nurse ward
leaves on average **13 shift-slots understaffed, 12.8 shifts with no senior nurse, and 36 approved day-off requests
ignored** (5 synthetic instances; generator parameters in `trackE/nurse.py::GEN`, demand ≈ 97% of contracted capacity,
as on a busy ward). Those are patient-safety and staff-retention problems, not just inconvenience.

### 1.2 Formulation
Decision variables x[n,d,s] ∈ {0,1} (nurse n works shift s on day d).

| Hard constraints | Soft objective (minimise, weights in brackets) |
|---|---|
| H1 ≤ 1 shift per nurse per day | S1 understaffing, per missing nurse [100] |
| H2 no Morning/Evening the day after a Night (rest) | S2 shift without a senior nurse [60] |
| H3 contract: full-time ≤ 10, part-time ≤ 6 shifts / 14 days | S3 day-off request worked [10] |
| H4 ≤ 3 consecutive nights | S4 night fairness: max − min nights among full-timers [5] |
| H5 approved leave never worked | S5 overstaffing, per extra nurse [2] |

The ILP linearises S1/S5 with `under[d,s] − over[d,s] = demand − staff`, S2 with a binary `nosenior[d,s] ≥ 1 − Σ seniors`,
and S4 with `nmax ≥ nightsₙ ≥ nmin`. Success metric = total penalty from one **independent checker** (`evaluate()`), plus
hard violations (must be 0) and runtime.

### 1.3 Strategy selection

| Method | Variable type | Exact? | Size it handles here | Hardware | Parallel? | Role |
|---|---|---|---|---|---|---|
| Manual rule (baseline) | binary | no | any, instant | — | no | what wards do now |
| Greedy with scoring | binary | no | any, < 0.2 s | 1 core | no | fast first roster |
| Simulated annealing | binary | no | any, ≈ 2 s | 1 core | could run chains in parallel | improve greedy |
| ILP (HiGHS branch-and-cut) | binary + continuous | **yes** (proves optimality) | 3,500 vars / 4,400 rows in ≈ 9 s | 1 core | solver-internal | best quality |
| Rejected: gradient methods | — | — | — | — | — | variables are discrete; relaxation + rounding breaks H2/H4 |
| Rejected: QAOA / annealer | — | — | — | — | — | 560–3,360 binary variables far exceed simulators and embeddable QUBO size (Track I) |

### 1.4 Results (mean of 5 instances per ward size; full table in the summary)

| Nurses | Manual | Greedy | Sim. annealing | ILP (time) |
|---|---|---|---|---|
| 10 | 1,996 | 714 | 291 | **147** (0.8 s) |
| 20 | 2,458 | 448 | 186 | **37** (1.7 s) |
| 40 | 4,179 | 911 | 413 | **8** (3.5 s) |
| 80 | 7,659 | 1,944 | 1,205 | **52** (9.3 s) |

The ILP proved optimality on all 20 instances. For the 20-nurse ward it removes all understaffing and all no-senior
shifts and honours all but 3.6 of ~36 requests, with nights spread within 0.2 shifts. Annealing roughly halves greedy's
penalty but stalls because its moves cannot trade one nurse's shift for another's across days.

**Correctness.** The ILP's objective is asserted equal to the independent checker's cost on every instance; every
method has 0 hard violations (the checker found a bug in the annealing feasibility test — failure #8).

**Sensitivity and limits.** With demand +25% (flu season) the manual roster is 47 slots short and the ILP still
46 short: optimisation cannot create capacity. The model's value then is to quantify the staffing gap. Limits: synthetic
data; equal-weight objective (weights are a management decision); no individual skill matrix beyond senior/junior;
real wards need 4–6-week horizons (≈ 2–3× variables; still within ILP reach given the scaling observed).

---

## 2. Track A — Continuous optimisation
**Objective.** Compare GD, SGD, mini-batch SGD, Momentum, NAG, AdaGrad, RMSProp, Adam, Newton and Gauss–Newton on
landscapes chosen to expose their differences, and on an ML task. All optimisers are written from scratch
(`trackA/optimizers.py`); learning rates tuned by grid search with the same iteration budget for every method.

**Findings.**
* *Strongly convex quadratic (κ = 50):* NAG 67 iterations to 1e-6, Momentum 132, Adam 125, GD 214, Newton 1. GD with
  lr = 1/L stays below the theoretical (1 − μ/L)^2k curve at every step (asserted).
* *Convex but not strongly convex (Σx⁴):* GD's log-log slope is −1.95 (theory −2) — sublinear, as predicted when
  curvature vanishes at the optimum.
* *Non-convex Rosenbrock:* Newton 5 iterations, Momentum 394; GD, AdaGrad and RMSProp do not reach 1e-6 in 5,000.
* *Saddle point:* first-order methods escape to the minima; **plain Newton converges to the saddle** (Hessian
  eigenvalues −2, 2), saddle-free Newton (|H|) escapes. *Local minima:* 52–62% of starts reach the global minimum;
  momentum helps slightly.
* *Noise:* with gradient noise σ = 2, a decaying step 1/(μ(k + k₀)) reaches f ≈ 0.002, ten times better than constant steps.
* *ML (logistic regression):* at equal epochs, mini-batch and adaptive methods reach f − f* ≈ 2–4e-4; full-batch GD
  2.4e-3. Test accuracy ≈ 0.96 for all — optimisation precision beyond this does not change generalisation.
* *Backprop* for a 65-parameter MLP matches finite differences to 4.5e-10; the same gradient is fed to four optimisers.
* *Curvature:* exact Newton, Newton-CG with Hessian-vector products (112 HVPs total) and natural gradient all reach
  1e-8 in 7 iterations vs 1,638 for GD; diagonal preconditioning 96 (after step tuning), top-5 low-rank Newton 27.
  The model Fisher equals the Hessian for logistic loss; the empirical Fisher differs by 74% and converges much worse.
* *Gauss–Newton* fits a 3-parameter exponential in 7 iterations (Adam 273, GD 505).

**Trade-off.** Second-order methods win per iteration but cost O(d³) per step (or many HVPs); for large models
diagonal/low-rank/Adam-style approximations are the practical middle ground.

## 3. Track B — Discrete optimisation
* *Knapsack (n = 20–200):* DP, ILP and branch-and-bound always agree (asserted); greedy is within 1.1% at n = 20 and
  0.03% at n = 200. DP is fastest here (capacity is small); ILP has the largest fixed overhead.
* *Makespan:* LPT is 2.5–4% above optimal and always within Graham's 4/3 − 1/(3m) bound; our B&B needs ~970,000 nodes
  (2.8 s) at n = 20 versus 0.14 s for the ILP — exponential growth of plain enumeration.
* *Max-Cut QUBO/Ising:* the energy of every one of the 2⁸ assignments is identical in the direct cut, QUBO and Ising
  forms (mismatch 0.0). Local search finds the optimum 55% of the time, SA 100%. This instance is reused in Tracks C/I.
* *Penalty QUBO (choose exactly 6 features):* P ≤ 0.5 returns infeasible subsets of size 3–5; P ≥ 1 returns the
  constrained optimum — consistent with the rule "P must exceed the largest single gain (0.78)".

## 4. Track C — Quantum optimisation (simulated)
* *QAOA on Max-Cut:* expected-cut approximation ratio 0.72–0.84 depending on depth and optimiser (COBYLA cheapest at
  ~50–200 circuit evaluations; parameter-shift GD 2,000–6,000). **The classical baseline wins**: SA finds the optimum
  every time, local search averages 0.94. QAOA's best *sampled* cut is optimal, but the expected cut is not.
* *Shots:* estimator std follows 1/√shots (std·√shots ≈ 1.6–2.2 constant); training with 64 shots drops the ratio to 0.69.
* *Noise:* the ratio falls from 0.84 (noiseless) to 0.72 at 5% two-qubit depolarising error — approaching random (0.64).
* *VQE (H₂):* QNG reaches chemical accuracy in 4 iterations, parameter-shift GD 18, COBYLA 45; SPSA only 2/5 runs in 100
  iterations (cheapest per step: 2 evaluations regardless of parameter count).
* *Barren plateau:* gradient variance of a global cost decays ≈ 2^(−0.89n) with qubits; a local cost decays much
  more slowly — cost-function design matters more than the optimiser.

Simulation vs hardware: everything is exact or sampled simulation; the noise model is synthetic (Track I uses a
device-calibrated model).

## 5. Track D — Submodular, parallel and distributed
* *Submodular:* weighted coverage passes 3,000 random diminishing-returns tests (0 violations), so greedy's (1 − 1/e)
  guarantee applies; observed greedy/OPT ≥ 0.94. Lazy greedy gives identical answers with 494 instead of 7,225 oracle
  calls. A complementary (pair-bonus) objective fails the test (209 violations) and no guarantee is claimed.
* *Load balancing (measured, pthreads, heavy-tailed sorted tasks):* static block split reaches only 1.1× on 2 cores
  (one thread gets the largest tasks, imbalance 1.77); cyclic, central dynamic queue and work stealing reach ≈ 2.0×.
  Simulation to 16 workers: block 2.8×, stealing 14.7×.
* *Amdahl/Gustafson:* the Karp–Flatt serial fraction from 2 cores is within timing noise of zero (repeat runs gave
  −0.016 and +0.016), so it cannot be extrapolated. For an assumed serial fraction of 5%, Amdahl caps 16 cores at 9.1×
  while Gustafson's scaled speedup is 15.3× (**model predictions, not measurements** — only 2 cores were available).
* *Locality:* loop order ikj is 1.4–4.4× faster than ijk (sequential access of B's rows); 64-blocking helps less here.
* *Synchronisation:* a mutex per increment makes 2 threads 3–4× *slower* than 1 (lock ping-pong); atomics are 3× faster than the mutex;
  thread-local accumulation removes the cost entirely.
* *Communication:* with 1 KB–10 MB arguments and tiny tasks, a 2-process pool is 2–4,000× slower than serial —
  pickling dominates; only coarse tasks with small payloads benefit.
* *Data/model/pipeline parallelism (NumPy MLP):* data-parallel gradients equal the full-batch gradient (7e-13) with
  1.69× on 2 processes; tensor-parallel outputs are exact (1.18×); a 2-stage process pipeline is slower than serial
  (queue serialisation), while the GPipe model shows the bubble (S−1)/(M+S−1) shrinking with more micro-batches.
* *FedAvg (10 clients):* more local steps E cut communication (IID: 37 → 15 rounds from E = 20 → 50; E ≤ 5 never reach
  the target in 60 rounds). Label-skewed clients plateau ~15× higher (client drift).
* *Multi-task:* GradNorm gives the lowest mean relative loss; MGDA's min-norm direction favours one task and leaves the
  large-scale task worse; equal weighting is in between. No method dominates on both tasks.

## 6. Track F — Compiler optimisation
**Build.** MiniC: lexer, recursive-descent parser, three-address IR, an IR virtual machine (instruction count and a
cycle model), and a C back end compiled with `gcc -O0` (so gcc does not redo our optimisations).
**Optimisations (6):** global constant propagation and folding (dataflow over the CFG, including constant branches),
copy propagation, local CSE (value numbering), liveness-based DCE + unreachable code, loop-invariant code motion over
natural loops, strength reduction/algebraic simplification. Plus Chaitin–Briggs and ILP register allocation.

**Results.** Dynamic IR instructions fall 20–60% on all five benchmarks and static size by 30–60%; outputs are
identical in the VM and natively. Native speedups under gcc -O0 range from 0.90× to 1.95× — `sum_invariant` is
*slower* (0.84–0.90× across runs) even though it executes 40% fewer IR instructions, because at -O0 every variable is a stack slot and the loop
is limited by the store→load chain that both versions share (a measurement-validity lesson). Ablation: LICM is the
single most valuable pass; removing DCE costs most within the pipeline.

**Formulated decisions.**
1. *Pass ordering:* choose 4 passes from 6 (1,296 sequences) to minimise total cycles. Exhaustive search finds
   753,981 cycles; greedy finds 777,888 (3% worse) with 24 instead of 1,296 evaluations. The textbook order
   (constprop, copyprop, cse, sr) is poor here (892,015) because it omits LICM.
2. *Register allocation:* minimise loop-weighted spill cost subject to K-colouring. On our programs Chaitin–Briggs is
   already optimal (ILP proves it). On random interference graphs Chaitin spills 2–6× more cost; the ILP costs
   0.6–4.5 s vs 0.2 ms.

**Correctness:** differential fuzzing — 1,000 random programs × 3 pass pipelines + register allocation, 0 mismatches.

## 7. Track G — xv6 operating system (QEMU, RISC-V)
**Features (patch `trackG/xv6-a8.patch` against upstream `e93dabe`):**
1. **MLFQ scheduler** (runtime-selectable; round robin remains the default): 3 levels with quanta 1/2/4 ticks, demotion
   on quantum exhaustion, priority boost every 50 ticks, preemption when a higher-priority process becomes runnable.
   *OS concept:* scheduling policy, the turnaround/response trade-off, starvation.
2. **Copy-on-write fork** (runtime-selectable): shared read-only pages with a `PTE_COW` bit, per-page reference counts
   in the allocator, a store-page-fault handler, and COW-aware `copyout`. *OS concept:* virtual memory, page faults,
   lazy work.
3. Syscalls `setsched`, `waitstat` (per-process creation/first-run/exit/run ticks), `setcow`, `memstat`.

**Results (1 CPU, 3 fresh boots per configuration; ticks ≈ 0.1 s).** *COW:* fork+exit with a 2 MB heap takes 1.1 ms
instead of 17.9 ms (16×) and copies 0 instead of 10,340 pages per 20 forks; 8 live children use 57 instead of 4,192
pages (−98.6%). The trade-off is real: when the child writes 100% of the heap, COW is 1.9× *slower* (37 vs 19 ms — a
fault plus a copy per page instead of one bulk copy); break-even is ≈ 50% of pages written.
*MLFQ:* short jobs arriving behind long jobs (W3) finish in 0.1 instead of 2.1 ticks with zero response time;
interactive jobs sharing the CPU with hogs (W2) finish 41% sooner (13.0 vs 22.0 ticks). The cost: an all-CPU-bound
batch (W1) has 11% worse average turnaround (12.6 vs 11.3) because demoted jobs wait behind each other. Before the
preemption fix (failure #9) W2 was 51% *worse* than round robin (`results/trackG/sched_bench_BEFORE_preemption_fix.json`).

**Correctness:** an isolation test (parent/child/grandchild and a pipe write into a shared page never leak);
**`usertests`: ALL TESTS PASSED (65 tests, 3 CPUs) both with COW + MLFQ switched on and with both off**
(`trackG/usertests_*.log`).
The first full run exposed a lock-order deadlock in the statistics code (failure #13) that the 1-CPU benchmarks never hit.

## 8. Track H — DBMS (PostgreSQL 16, TPC-H SF 0.1)
**Workload:** 8 weighted OLTP-style lookups of an order-management application + TPC-H Q3, Q5, Q6, Q10, Q12, Q14.
**Formulation:** choose x_i ∈ {0,1} over 16 candidate indexes (from WHERE/JOIN/ORDER BY columns) to minimise
Σ_q w_q·cost(q | indexes) subject to Σ size_i·x_i ≤ B. Cost oracle: PostgreSQL's planner with HypoPG hypothetical
indexes ("what-if"), so candidates are evaluated without building them.
**Methods:** default (primary keys only), greedy (best what-if saving per byte), ILP (atomic-benefit model, HiGHS),
and "all candidates" as the unlimited-storage reference.
**Results (measured after building each configuration; median of 15 warm runs, two independent measurement runs).**
The weighted workload falls from ≈ 3.0 s (primary keys only) to ≈ 0.49 s with an 11 MB index budget (≈ 6×), close to
the 48 MB "all candidates" result. At the tightest budget (10%) the ILP is ≈ 10% faster than greedy in both runs
(2.22 vs 2.47 s) and has the lower what-if cost: ratio-greedy spends the budget on small indexes, the ILP buys the one
large `lineitem(l_partkey)` index. At 25% both choose the same indexes, and at 50% their difference is within run-to-run
noise (±10%). An earlier 5-run measurement showed the 10% ordering *reversed* — the reason for 15 runs and two repeats
(`results/trackH/index_selection_run1.json`). Lesson: planner cost is a good but imperfect oracle for wall-clock time.
**Correctness:** every configuration returns the same row multiset for every query (exact order where `ORDER BY` defines
it — failure #12 shows why that distinction matters). `EXPLAIN ANALYZE` plans are saved (Seq Scan → Bitmap Index Scan).

## 9. Track I — Quantum algorithms and QML vs classical (simulated)
All circuits are hand-written: Deutsch–Jozsa (1 query vs 17 classical worst case, n = 5), Bernstein–Vazirani (1 vs 6),
Grover, QFT (matches the DFT matrix to 4e-15), QPE (error halves with each counting qubit), Shor (factors 15 for
a = 7, 2, 4, 11, 13), and quantum annealing (exact adiabatic simulation of the Track B Max-Cut: P(optimum) 0.01 at
T = 0.5 → 0.997 at T = 40, governed by the minimum spectral gap).

* **Grover vs linear search:** query counts follow √N vs N/2 (25 vs ~512 at N = 1,024), but *simulating* Grover
  takes longer than the classical search at every size and grows exponentially — a query advantage is not a
  wall-clock advantage without hardware. On the FakeManilaV2 device noise model, Grover (n = 3, 37 two-qubit gates
  after routing) succeeds with P ≈ 0.55 instead of 0.94.
* **Annealing hardware limits:** our 8-node Max-Cut embeds easily, but complete graphs need chains of physical qubits;
  larger K_n need many qubits per logical variable or fail on Chimera — dense QUBOs (e.g. Track E) do not fit.
* **QML vs classical (moons, 5 splits):** the variational classifier (12 parameters) reaches 93.3 ± 7.0% test accuracy —
  better than logistic regression (88.3%) and a small MLP (86.7%) but below an RBF-SVM (97.2%), which trains in about a
  millisecond versus 22 s and ~126,000 circuit evaluations. The quantum-kernel SVM reaches 77.8%. Net: no quantum
  advantage on this classical low-dimensional data; on hardware each evaluation would also cost many shots.
* **Contrast with Track D:** parallel speedups are *measured* today (≈ 2× on 2 cores) and bounded by Amdahl; quantum
  speedups here are *theoretical query counts*, and simulation is exponentially slower.

## 10. Track J — Software engineering: test-suite minimisation (more-itertools)
**Subject:** `more-itertools` (7,282 library lines, 769 tests, ~21,000 subtests; public OSS project).
**Profile:** full suite 17.7 s wall. Per-test times come from the coverage-instrumented run and are used as relative
weights. **One test (`PrimeFunctionTests.test_primes`) accounts for 65% of the time**, the next one 13%.
**Formulation:** weighted set cover — x_t ∈ {0,1}, minimise Σ time_t·x_t such that every line covered by the full suite
stays covered. **Methods:** greedy (most new lines per second), ILP (HiGHS, exact), random subset of the same size.
**Results:** both minimisers keep 100% line coverage with ~65% fewer tests (ILP 266, greedy 291 of 769), whereas a
random subset of 266 keeps only 69.5% coverage. But **measured run time does not fall** (18–21 s vs 17.7 s): the slow
tests cover lines no other test covers, so every coverage-preserving suite must include them. The formulation optimised
the test *count*, while the bottleneck is one irreplaceable slow test — the next target is that test itself (or a
coverage criterion at function level). *Fault detection:* of 40 AST mutants, 28 are killed by the full suite; the
reduced suites kill 78.6% of those, a random subset 67.9% — 100% line coverage does not preserve assertions.
*Prioritisation:* ordering by additional coverage per second gives APFD 0.78 vs 0.51 for the original order.

---

## 11. Answers to the final questions (Section 11)

1. **Process and evidence.** Nurse rostering (Track E): the manual-style roster leaves 13 slots understaffed, 12.8
   shifts without a senior nurse and ~36 requests ignored per 20-nurse ward fortnight.
2. **Formulation.** Binary x[n,d,s]; five hard constraints; weighted soft penalties linearised with auxiliary
   variables (§1.2). Each domain track states its own formulation (index selection = knapsack-like ILP, pass ordering
   = combinatorial search, register allocation = weighted colouring ILP, test minimisation = weighted set cover).
3. **Methods and why.** Discrete, constrained, mid-sized problems → ILP for proven optimality, greedy/annealing for
   speed (Tracks B, E, F, H, J). Smooth objectives → gradient and curvature methods (A). Monotone submodular →
   greedy with a checked guarantee (D). Quantum methods were tested where they apply (C, I) and rejected for Track E.
4. **Optimised vs baseline.** Rostering penalty −98.5% (20 nurses, ILP); compiler IR instructions −20 to −60%;
   COW fork 16× faster and 98.6% less memory; DBMS workload ≈ 6× faster; MLFQ interactive turnaround −41% and short-job
   turnaround 2.1 → 0.1 ticks; test suite −65% tests at equal coverage (but not faster).
5. **Trade-offs, failures, limits.** ILP time grows with size; COW loses when children write most pages; MLFQ needs
   preemption and boosting; a lower planner cost need not be faster; coverage-preserving minimisation does not cut time
   when slow tests are irreplaceable; MLFQ costs CPU-bound batches 11%; QAOA and QML lost to the best classical
   baselines; fifteen AI-generated bugs (failure log).
6. **Correctness and reproducibility.** Independent checkers and assertions (exact methods agree; ILP = checker cost),
   differential fuzzing, usertests on 3 CPUs, identical SQL result sets, gradient checks, fixed seeds, `run_all.sh`,
   unit tests, and auto-generated tables.
7. **Changing scale/data/hardware.** Rostering: longer horizons → ILP with time limits or rolling horizons;
   very large wards → annealing/large-neighbourhood search seeded by greedy. DBMS: larger workloads → greedy with
   what-if calls is cheaper than enumerating ILP benefits. Parallel: Amdahl predicts diminishing returns past ~16
   cores for our serial fraction. Quantum: only fault-tolerant hardware could turn query advantages into speed.
8. **Compiler / xv6 / DBMS concepts.** Compiler: dataflow analysis, dominators/natural loops, interference graphs —
   correctness shown by fuzzing. xv6: scheduling policy (starvation, preemption) and virtual memory (page faults,
   reference counting, kernel `copyout`) — correctness by usertests and an isolation test. DBMS: access paths and the
   cost-based optimiser — correctness by comparing result multisets.
9. **Quantum vs parallel.** Quantum algorithms give provable query reductions (DJ, BV, Grover) but, simulated, run
   slower; the variational classifier beat linear and small-MLP baselines but not an RBF-SVM, at ~10⁵ circuit evaluations; noise and barren plateaus limit scaling. Parallel computing
   gave measured ≈ 2× speedups on 2 cores, limited by load imbalance, synchronisation and communication.
10. **AI tools.** Claude Opus 5.5 for everything (register in `docs/AI_TOOLS.md`). Fifteen AI errors were found and
    fixed, including a kernel deadlock, a missing MLFQ rule, an infeasible ILP from a data-join bug, and an inverted
    classifier. The second-tool comparison is to be completed by the student (`docs/TOOL_COMPARISON.md`).

## 12. Limitations and next steps
* Only 2 CPU cores: parallel scaling beyond 2 is simulated or predicted.
* No real quantum hardware; noise models stand in for devices.
* Track E data are synthetic; next step is a real ward's demand and leave data, and a 4–6-week horizon.
* Track G benchmarks use one CPU for comparability; multi-CPU MLFQ needs per-CPU queues.
* Track H uses SF 0.1; a larger scale factor would increase index benefits and build costs.
* Track J should next target the slow, irreplaceable tests directly.
