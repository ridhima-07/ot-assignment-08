# Chapter 6 coverage checklist (Section 5)

"Demo" = how the method is demonstrated: **Impl** = implemented from scratch in this repo, **Lib** = library-based
experiment, **Sim** = simulation. Numbers are in `docs/RESULTS_SUMMARY.md` (generated from `results/`).
⚠ = please check against the exact syllabus taught in class and add/remove rows as needed (Section 5 last line).

## Foundations
| Topic | Demo | Code | Experiment / result |
|---|---|---|---|
| ML as optimisation (parameters, loss, regularisation) | Impl | `trackA/ml_optim.py::a6_regularisation` | L2 strength sweep 0 → 1: weight norm, train/test accuracy |
| Continuous vs discrete optimisation | Impl+Lib | `trackA/ml_optim.py::a11_continuous_vs_discrete` | Exhaustive 4-of-12 subset (discrete) vs Lasso relaxation (continuous): different subsets, higher SSE for relaxation |
| Convex vs non-convex landscapes | Impl | `trackA/landscapes.py::a1_convergence`, `ml_optim.py::a8_backprop` | quadratic/quartic vs Rosenbrock; MLP loss barrier between two minima |
| Local minima, global minima, saddle points | Impl | `landscapes.py::a2_saddle, a3_local_minima` | Hessian-eigenvalue classification of final points; basin statistics from 105 starts |
| Basic quantum computing concepts; role of optimisation in QML | Sim | `trackC/quantum_opt.py`, `trackI/quantum_algorithms.py` | VQE/QAOA/VQC are classical optimisation loops over circuit parameters |

## Continuous optimisation
| Method | Demo | Code | Result |
|---|---|---|---|
| Gradient descent | Impl | `trackA/optimizers.py::GD` | obeys (1−μ/L)^2k bound (asserted) |
| SGD, mini-batch SGD | Impl | `ml_optim.py::a7_optimizers` | batch 1 / 32 / full, 5 seeds, same epoch budget |
| Momentum | Impl | `optimizers.py::Momentum` | A1, A7 |
| Nesterov (NAG) | Impl | `optimizers.py::NAG` (look-ahead gradient) | fastest first-order on quadratic (67 iters) |
| AdaGrad, RMSProp, Adam | Impl | `optimizers.py` | A1–A7; RMSProp constant-step plateau explained in `tests/` |
| Newton | Impl | `optimizers.py::Newton`; A9 | 7 iterations to 1e-8; converges to the saddle in A2 |
| Gauss–Newton (and Levenberg–Marquardt) | Impl | `optimizers.py::GaussNewton`; A10 | 7 iterations vs 273 (Adam) / 505 (GD) |
| Backpropagation (distinct from optimiser) | Impl | `ml_optim.py::MLP.backprop` | finite-difference check 4.5e-10; same gradient fed to 4 optimisers |
| Fisher information / natural gradient | Impl | A9 | model Fisher = Hessian for logistic loss; empirical Fisher differs (0.74) |
| Diagonal and low-rank curvature | Impl+Lib | A9 (diag preconditioner sweep; Lanczos via `scipy eigsh` on HVPs) | 96 vs 1,638 iterations; Lanczos top-5 = exact |
| Hessian-vector products | Impl | A9 (`logreg_hvp`, finite-difference HVP), Newton-CG | 112 HVPs to converge |
| Convergence: convex, strongly convex, non-convex | Impl | A1 | linear (strongly convex), log-log slope ≈ −2 (quartic), Rosenbrock |
| Step-size / optimiser tuning | Impl | A4 `A4_stepsize.png` | divergence above 2/L for GD |
| Noisy objectives | Impl | A5 | decaying 1/(μ(k+k₀)) step beats constant steps |

## Discrete optimisation
| Method | Demo | Code | Result |
|---|---|---|---|
| Greedy | Impl | `trackB/discrete.py` (knapsack ratio greedy, LPT), Track E greedy, Track H greedy, Track J greedy | gaps vs exact reported |
| Integer linear programming | Lib (HiGHS) | knapsack, makespan, Max-Cut, nurse rostering, register allocation, index selection, test minimisation | 7 different ILP models |
| Branch-and-bound | Impl | `knap_bnb` (LP bound), `makespan_bnb` (symmetry breaking) | node counts; exponential blow-up at n=20 |
| Dynamic programming | Impl | `knap_dp` | exact, pseudo-polynomial O(nC) |
| Feature selection / combinatorial apps | Impl | A11, B4 | exhaustive, Lasso, QUBO |
| QUBO and Ising (binary→spin, penalty terms) | Impl | `maxcut_qubo`, `qubo_to_ising`, `b4_penalty` | 0 energy mismatch over 256 assignments; penalty sweep |
| Max-Cut, feature selection (as QUBO) | Impl | B3, B4; QAOA in C1; annealing in I7 | |

## Quantum optimisation and algorithms (all simulated)
| Method | Demo | Code | Result |
|---|---|---|---|
| QAOA | Sim (hand-built circuit) | `trackC/quantum_opt.py::QAOA` | p = 1..3 vs classical baselines |
| VQE | Sim | `c4_vqe` (H₂) | chemical accuracy |
| PQC / QNN | Sim | VQE ansatz; `trackI::train_vqc` (data re-uploading) | |
| Parameter-shift rule | Impl | `QAOA.grad_param_shift`, `vqe_grad`, VQC training | vs finite differences 1e-9 |
| SPSA | Impl | `spsa()` | C1, C4 |
| Quantum natural gradient | Impl | `qfi_metric` (Fubini–Study) | 4 iterations to chemical accuracy |
| Hybrid quantum–classical | Sim | all variational loops | |
| Deutsch–Jozsa, Bernstein–Vazirani | Sim (own circuits) | `trackI` I1, I2 | 1 query vs 2^(n−1)+1 / n |
| Grover | Sim (own circuit) + noise model | I3 | query counts, gate counts, wall-clock, FakeManila |
| QFT, QPE | Sim (own circuits) | I4, I5 | DFT match 4e-15; error vs counting qubits |
| Shor | Sim (own QPE + mod-15 multipliers) | I6 | factors 15 for 5 bases |
| Quantum annealing | Sim (exact adiabatic evolution) | I7 | P(opt) vs anneal time; spectral gap |
| Variational classifiers, quantum kernels | Sim | I8 | vs logistic / RBF-SVM / MLP |
| Barren plateaus | Sim | C5 | global vs local cost variance |
| Finite-shot cost, hardware noise | Sim | C2, C3, I3 | 1/√shots law; ratio vs noise |
| Limits of embedding QUBOs on annealers | Lib (minorminer) | I7 | chain lengths on Chimera / Pegasus; K_n failures |

## Submodular optimisation
| Topic | Demo | Code | Result |
|---|---|---|---|
| Diminishing returns + testing assumptions | Impl | `trackD/submodular.py::test_submodular` | 0 violations (coverage) vs 209 (complementary) |
| Greedy selection; checking the (1−1/e) guarantee conditions | Impl | `greedy`, `lazy_greedy`, exhaustive comparison | guarantee claimed only where the test passes |

## Parallel program optimisation
| Topic | Demo | Code | Result |
|---|---|---|---|
| Load balancing, makespan, work stealing | Impl (C pthreads) + Sim | `par_bench.c sched`, `parallel.py::p7_sim` | block vs cyclic vs central vs stealing |
| Locality | Impl (C) | `par_bench.c locality` | ijk vs ikj vs blocked |
| Synchronisation | Impl (C) | `par_bench.c sync` | mutex / atomic / local |
| Communication overhead | Impl | `parallel.py::p5_comm` | pickling cost dominates small tasks |
| Data, model, pipeline parallelism | Impl | `parallel.py::p6_parallelism` | equality checks; GPipe bubble model; measured pipeline |
| Speedup, efficiency, Amdahl, Gustafson | Impl | `p1_p2_sched` | Karp–Flatt fit; predictions labelled as such |

## Strategy selection, distributed and multi-task
| Topic | Demo | Code | Result |
|---|---|---|---|
| Choosing methods by variables, constraints, size, hardware, convergence, communication | Report | `REPORT.md` §E strategy table + per-track "why this method" | |
| FedAvg | Impl | `trackD/distributed.py::fedavg` | local steps vs rounds; IID vs non-IID |
| MGDA | Impl (2-task closed form) | `mt_train(..."MGDA")` | |
| GradNorm | Impl | `mt_train(..."GradNorm")` | |

## Domain tracks also using Chapter 6 methods
| Track | Methods |
|---|---|
| E nurse rostering | greedy, simulated annealing, ILP |
| F compiler | greedy vs exhaustive pass ordering; greedy (Chaitin) vs ILP register allocation |
| G xv6 | MLFQ scheduling (online heuristic), COW (lazy evaluation) |
| H DBMS | greedy vs ILP index selection over a what-if cost oracle |
| J software engineering | greedy vs ILP weighted set cover; greedy prioritisation |
