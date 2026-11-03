# 5-minute demonstration script

Goal: show **problem → method → evidence of improvement → lesson**, live, using commands that finish in seconds.
Open `results/trackE/E_rosters.png`, `results/trackG/G_cow.png` and `results/trackH/H_index_selection.png` beforehand.

## 1. The problem (1 min) — Track E
* Show `E_rosters.png`: left, a hand-made roster (dark gaps = understaffing, nights piled on a few nurses); right, the ILP roster.
* Say: *"A 20-nurse ward, 14 days, 3 shifts. The manual roster leaves 13 shifts short-staffed and 13 without a senior nurse."*

## 2. The method (1.5 min)
```bash
python3 -c "
import sys; sys.path.insert(0,'.')
from trackE.nurse import *
I = instance(20, 0)
for name, x in [('manual', manual_baseline(I)), ('greedy', greedy(I)), ('ILP', ilp(I, 30)[0])]:
    e = evaluate(I, x); print(f'{name:7s} penalty={e[\"cost\"]:5d} understaffed={e[\"understaffed\"]} no_senior={e[\"no_senior_shifts\"]} hard_violations={e[\"hard_violations\"]}')"
```
Explain: binary variables x[n,d,s], hard constraints, weighted penalties; ILP proves optimality in about a second;
the same independent checker grades all methods.

## 3. Evidence in other domains (1.5 min)
```bash
python3 trackF/fuzz.py 100                                   # compiler: 0 output mismatches
python3 trackG/qemu_run.py external/xv6-riscv "cowbench 0 1" "cowbench 1 1"   # xv6: eager vs copy-on-write fork
```
Show `G_cow.png` (COW wins when children write little, loses at 100%) and `H_index_selection.png` (≈ 6× faster workload with 11 MB of indexes).

## 4. Lessons (1 min)
* Optimisation needs a **formulation and a fair baseline**, not just code (Track J: fewer tests, same run time).
* **Verify independently**: the checker, fuzzing and `usertests` caught 15 AI-generated bugs, including a kernel deadlock.
* Honest negatives count: QAOA and the quantum classifier lost to classical baselines on these problems.
