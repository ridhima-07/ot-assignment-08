# Failure log — AI-generated code that was wrong, unsafe, inefficient or misleading

Every entry below happened during development with **Claude Opus 5.5** (see `AI_TOOLS.md`). Each was caught by a
verification step that is still in the repository, so it can be re-checked. The assignment asks for at least three;
eleven are recorded because each taught something different.

| # | Track | What the AI produced | How it was caught | Fix | Lesson |
|---|---|---|---|---|---|
| 1 | A | Evaluated Fisher-vs-Hessian at **w = 0**, reporting that the empirical Fisher equals the Hessian (rel. diff 5.6e-16). | The result was "too good": the empirical Fisher should differ from the Hessian. At w = 0 every sigmoid is 0.5, so (s−y)² = s(1−s) = 0.25 exactly. | Evaluate curvature at w = ½·w\* (`trackA/ml_optim.py`, `a9_curvature`). Rel. diff is now 0.74. | A test point can make two different quantities coincide. Check results that look too perfect. |
| 2 | A | Diagonal-Hessian preconditioning reported as "did not converge" from a single step size (0.5). | `null` in the iterations table with no context. | Step-size sweep {0.5…0.02}; reports all, best = 96 iterations vs 1,638 for GD. | Do not judge a method from one untuned setting (Section 10 fairness). |
| 3 | B | Penalty-QUBO demo with k = 3: every penalty weight, including P = 0, "satisfied" the constraint. | The sweep showed no effect at all. | Unconstrained optimum already had 3 features; switched to k = 6 so the constraint binds. Now P < 1 is infeasible, P ≥ 1 matches the constrained optimum. | A demonstration must be able to fail, or it demonstrates nothing. |
| 4 | C | QAOA shot-based energy bound parameters by object identity; the measured circuit had fresh `ParameterVector` objects → `KeyError`. | Crash after a 3-minute run. | Bind by parameter **name** (`QAOA._order`). | Qiskit parameters are identity-based; rebuilding a circuit creates new ones. |
| 5 | D | Multi-task validation set generated **different task vectors** (drawn from the same RNG as X), so every method looked worse than predicting the mean (rel. loss 1.7–1.9). | Relative loss > 1 is impossible for a trained model on the same task. | Task vectors fixed with their own seed (`mt_data`). | Separate the randomness that defines the *problem* from the randomness that samples *data*. |
| 6 | D | FedAvg reference optimum computed with fixed-step GD (lr = 1) on unscaled features; it diverged, giving **negative** sub-optimality. | Negative f(w) − f\*. | L-BFGS reference + assertion ‖∇f‖ < 1e-5, standardised features. | Always assert the reference solution is actually optimal. |
| 7 | D | FedAvg target "test accuracy within 1% of centralised" was reached in round 1 by every configuration. | No differences between settings → metric uninformative. | Use global training-loss sub-optimality ≤ 2e-3. Client drift on non-IID data now visible. | Pick a metric that can separate the methods. |
| 8 | E | `feasible_to_add` checked only the 3 nights **before** a new night shift. Fine for day-ordered greedy, wrong for annealing's random moves → rosters with 4 consecutive nights. | The independent checker `evaluate()` reported 0.4–1.0 hard violations for simulated annealing. | Count the whole run of consecutive nights in both directions. | Independent checkers catch bugs that the generating code "proves" impossible. |
| 9 | G | MLFQ without the preemption rule: a demoted CPU hog kept the CPU for its whole 4-tick quantum when an interactive job woke up. Interactive turnaround got **51% worse** than round robin (22 → 33 ticks). | Benchmark W2. | `sched_tick()` now yields when a higher-priority process is RUNNABLE. | "Compiles and boots" ≠ correct policy. An AI-written scheduler omitted a defining rule of MLFQ. |
| 10 | G | Kernel patch: `copyout` called with the wrong number of arguments (this xv6 revision has a `psz` parameter); inserted code also split `int` from `uvmcopy`'s name. | `-Werror` build failure. | Matched the tree's real signature. | AI tools assume the textbook xv6 API; read the actual tree. |
| 11 | I | Variational classifier predicted class 1 when ⟨Z⟩ < 0, but trained class 1 towards ⟨Z⟩ = +1 → **6.7% test accuracy** (every label inverted). | Accuracy far *below* chance on a 2-class problem. | Predict class 1 when ⟨Z⟩ > 0. | Below-chance accuracy means a sign/label bug, not a weak model. |

## Measurement pitfalls (not bugs, but misleading if reported naively)

* **Track D, 2-thread timings were initially ~1.0×** because the Track C simulation was occupying the second core. All
  parallel timings were re-measured on an idle machine.
* **Track F, `sum_invariant` runs 15% slower natively** after optimisation even though it executes 40% fewer IR
  instructions. At `gcc -O0` every variable lives on the stack; run time is set by the loop-carried
  store→load chain through `s` and `i`, which both versions share. IR instruction count is not run time.
* **Track F, Chaitin vs ILP** percentages look huge (up to +2200%) because optimal spill costs are tiny (e.g. 1 vs 13).
  Absolute costs are reported next to percentages.
