# Viva preparation

The viva requires explaining and **modifying code live without an AI tool**. For each track: the one-paragraph
explanation to give, the questions examiners are likely to ask, and a small change to practise beforehand
(commit each as `[manual] ...`).

---

## Track A — continuous optimisation
**Explain.** All optimisers share `step(x, g)`. Momentum keeps a velocity; NAG evaluates the gradient at the
look-ahead point `x − lr·β·v`; AdaGrad/RMSProp/Adam divide by a running root-mean-square of gradients; Newton solves
`H d = g`; Gauss–Newton replaces H by JᵀJ for least squares. Backprop (`MLP.backprop`) only *computes* the gradient
with the chain rule; the optimiser decides what to do with it.

**Likely questions**
* Why does plain Newton converge to the saddle? — It jumps to any point where ∇f = 0; it uses H⁻¹, and with a negative
  eigenvalue it moves *towards* the maximum along that direction. Saddle-free Newton uses |H|.
* Why is GD's rate (1−μ/L)? — Condition number κ = L/μ = 50; the slow direction shrinks by (1 − 1/κ) per step.
* Why does the empirical Fisher differ from the Hessian? — It uses observed labels (s−y)², the model Fisher uses the
  model's own label distribution s(1−s); they agree only in expectation (and coincidentally at w = 0, failure #1).
* Why does RMSProp plateau? — Normalised steps of size ≈ lr regardless of gradient size → oscillation.

**Practise:** add an `AdamW` class (decoupled weight decay) to `optimizers.py` and run it in `a7_optimizers`.

## Track B — discrete
**Explain.** Knapsack: greedy by value/weight (½-approx when combined with best single item), DP over capacity
O(nC), ILP via HiGHS, B&B with the fractional (LP) relaxation as the upper bound. Max-Cut: cut(x) = Σ w(xᵢ + xⱼ − 2xᵢxⱼ);
Ising via x = (1 − s)/2. Penalty: add P(Σx − k)²; P must exceed the largest gain from violating.

**Questions:** Why is DP "pseudo-polynomial"? Why did B&B explode at n = 20 for makespan? What does Graham's bound say?
**Practise:** change `b4_penalty` to "at most k" (inequality) using a slack variable.

## Track C — quantum optimisation
**Explain.** QAOA alternates exp(−iγC) (RZZ gates on edges) and exp(−iβB) (RX). The classical optimiser tunes γ, β.
Parameter-shift: for a gate exp(−iθP/2), ∂E/∂θ = [E(θ+π/2) − E(θ−π/2)]/2 exactly. SPSA perturbs all parameters at
once (2 evaluations per step). QNG preconditions with the Fubini–Study metric.

**Questions:** Why did QAOA not beat simulated annealing? (shallow p, 8 nodes is trivial classically; expected-cut
ratio vs best-sample). Why does gradient variance vanish for global cost? (concentration of measure in 2ⁿ dims).
**Practise:** change `c2_shots` to 8192 shots and confirm std halves vs 2048.

## Track D — submodular / parallel / distributed
**Explain.** Greedy's 1 − 1/e guarantee needs monotone submodular f and a cardinality constraint — so the code
*tests* diminishing returns before claiming it. Work stealing: idle threads take tasks from the top of a busy
thread's deque. Amdahl: S = 1/(s + (1−s)/p) (fixed problem); Gustafson: S = p − s(p−1) (problem grows with p).
FedAvg: clients run E local SGD steps, server averages weighted by data size. MGDA: min-norm combination of task
gradients; GradNorm: learns task weights so gradient norms track relative training rates.

**Questions:** Why is block splitting slow here? (tasks sorted by size → last thread gets the heaviest). Why is the
mutex version slower with 2 threads than 1? (lock ping-pong between cores). Why does non-IID FedAvg plateau? (client drift).
**Practise:** run `./par_bench sched 2 block` after shuffling `work[]` instead of sorting — imbalance disappears.

## Track E — nurse rostering (main case study)
**Explain.** Variables x[n,d,s]. Hard constraints H1–H5, soft penalties S1–S5 (see `trackE/nurse.py` header). The
ILP adds auxiliary `under`, `over`, `nosenior`, `nmax`, `nmin` variables to linearise the soft terms. Greedy fills
hard shifts first with a scoring rule; annealing improves it with feasibility-preserving moves; `evaluate()` is an
independent checker used for all methods (it caught failure #8).

**Questions:** Why ILP here and not in Track J's larger problems? (≈ 1,000–3,500 variables solve in seconds; strong LP
relaxation). What if demand exceeds capacity (+25% scenario)? (no method can staff it; the model shows *how much*
short — a staffing decision, not a scheduling one). How would you add weekends-off fairness? (another nmax/nmin pair).
**Practise:** add a hard constraint "at most 5 working days in any 7" and rerun `run_all(sizes=(20,))`.

## Track F — compiler
**Explain.** Lexer → recursive-descent parser → three-address IR (temps `tN`) → passes → VM (counts instructions,
cycle model) and C back end. Constant propagation is a forward dataflow over the CFG with lattice
{TOP, constant, NAC}; DCE uses backward liveness; LICM finds natural loops from back edges (n→h where h dominates n)
and hoists single-definition temporaries whose operands are not defined in the loop; it never hoists a possibly
trapping division. Register allocation: interference graph + Chaitin–Briggs or an ILP minimising loop-weighted spill cost.

**Questions:** Why must LICM avoid hoisting `x / y`? Why is native speed not improved for `sum_invariant`? (store→load
chain at -O0). Why is greedy pass ordering 3% off the exhaustive optimum?
**Practise:** add the strength reduction `x * 2 → x + x` and check with `python3 trackF/fuzz.py 300`.

## Track G — xv6
**Explain.** (1) **MLFQ**: 3 levels, quanta 1/2/4 ticks, demotion when a quantum is used up, boost to level 0 every 50
ticks, preemption when a higher-priority process becomes runnable (`sched_tick`, `scheduler` in `kernel/proc.c`).
(2) **Copy-on-write fork**: `uvmcopy_cow` maps parent pages read-only with the `PTE_COW` RSW bit into the child and
increments a per-page reference count (`kalloc.c`); a store page fault (`scause` 15) calls `cowfault`, which copies the
page (or reuses it if the count is 1). `copyout` also breaks COW because the kernel writes user memory directly.
Syscalls: `setsched`, `waitstat`, `setcow`, `memstat`.

**Questions:** Why does COW need reference counts? Why must `copyout` handle COW? Why did 100% writes make COW slower?
(a fault + copy per page instead of one bulk copy). What was the deadlock (failure #13) and why only on SMP? Why does
MLFQ need priority boost? (starvation) and preemption (failure #9)?
**Practise:** change the MLFQ quanta to {2, 4, 8} in `kernel/proc.c`, rebuild, rerun `schedbench 1 2`.

## Track H — DBMS
**Explain.** Index selection = choose x_i under a storage budget to minimise workload cost. Cost oracle = PostgreSQL's
own planner with HypoPG hypothetical indexes. Greedy adds the best saving-per-byte index repeatedly; the ILP assumes
per-index benefits are additive with at most one index per (query, table). Then indexes are built for real, the
workload timed, and result multisets compared.

**Questions:** Why does ILP beat greedy at the 10% budget? (knapsack: ratio-greedy fills the budget with small indexes).
Why can the ILP's choice be slower in reality at 50%? (planner cost ≠ wall time; additivity assumption). Read an
`EXPLAIN ANALYZE` plan (Seq Scan → Bitmap Index Scan).
**Practise:** add a query to `WORKLOAD` and a candidate index for it; rerun.

## Track I — quantum algorithms and QML
**Explain** each circuit in `trackI/quantum_algorithms.py`: DJ/BV use phase kickback onto |−⟩; Grover = oracle +
diffusion, ≈ π/4·√N iterations; QFT = Hadamards + controlled phases + swaps; QPE writes 2ᵏ-fold controlled-U phases
into a register and applies inverse QFT; Shor = QPE on modular multiplication + continued fractions.

**Questions:** Why does the query advantage not give a wall-clock advantage here? (statevector simulation costs 2ⁿ;
oracle is free classically). Why did the noisy device drop Grover from 0.94 to 0.55? (37 two-qubit gates after
routing). Contrast with Track D: parallel speedups are measured on real cores today.
**Practise:** run Grover for 2 marked items (oracle flips both) and recompute the optimal iteration count π/4·√(N/M).

## Track J — software engineering
**Explain.** Weighted set cover: keep every covered line, minimise total test time. Greedy picks the test with most
new lines per second (ln-factor approximation); ILP is exact. Validation: rerun the reduced suites and mutation testing.

**Questions:** Why can a suite with 100% line coverage miss mutants? (coverage ≠ assertions). What is APFD?
**Practise:** require each line to be covered **twice** (`>= 2` in `ilp_cover`) and compare the mutation score.
