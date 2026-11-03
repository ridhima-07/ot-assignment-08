"""Track F driver: measure the compiler optimisations and solve two compiler decisions as optimisation problems.

  F1 baseline vs optimised: dynamic instruction count, weighted cycles, static code size,
     compile time, native run time (generated C compiled with gcc -O0), output equality
  F2 ablation: each pass alone / default pipeline minus one pass
  F3 PASS ORDERING as optimisation: choose a sequence of L=4 passes (6^4 = 1296 options) minimising
     total weighted cycles over the benchmark suite. Exhaustive search (exact) vs greedy.
  F4 REGISTER ALLOCATION as optimisation: minimise loop-weighted spill cost subject to
     interference (graph K-colouring). Chaitin-Briggs greedy heuristic vs ILP (HiGHS).
  F5 differential fuzzing (1000 random programs)
"""
import sys, os, json, time, itertools, subprocess, tempfile, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trackF.minic import compile_to_ir, run_vm, static_size, to_c
from trackF.passes import optimise, PASSES, DEFAULT_PIPELINE, interference, color_chaitin, color_ilp, rewrite_with_registers
from trackF.fuzz import fuzz
from common.util import save_json, outdir, plt

T = "trackF"
HERE = os.path.dirname(os.path.abspath(__file__))
PROGS = ["sum_invariant", "cse_poly", "primes", "matmul", "fib_const"]
BIG = {"sum_invariant": ("n = 2000;", "n = 200000000;"), "cse_poly": ("n = 1500;", "n = 100000000;"),
       "primes": ("limit = 400;", "limit = 300000;"), "matmul": ("N = 20;", "N = 400;"),
       "fib_const": ("n = 3000 + 0 * m;", "n = 300000000 + 0 * m;")}


def src(p, big=False):
    s = open(os.path.join(HERE, "programs", p + ".mc")).read()
    if big: a, b = BIG[p]; assert a in s; s = s.replace(a, b)
    return s


def native(code, reps=3):
    d = tempfile.mkdtemp(); c = os.path.join(d, "p.c"); exe = os.path.join(d, "p")
    open(c, "w").write(to_c(code))
    subprocess.run(["gcc", "-O0", "-o", exe, c], check=True)
    ts = []; out = None
    for _ in range(reps):
        t0 = time.perf_counter(); r = subprocess.run([exe], capture_output=True, text=True, check=True); ts.append(time.perf_counter() - t0); out = r.stdout
    return statistics.median(ts), out, os.path.getsize(exe)


def f1_baseline_vs_opt():
    res = {}
    for p in PROGS:
        ir = compile_to_ir(src(p)); t0 = time.perf_counter(); opt = optimise(ir); tc = time.perf_counter() - t0
        a, b = run_vm(ir), run_vm(opt)
        irB = compile_to_ir(src(p, True)); optB = optimise(irB)
        tn0, out0, sz0 = native(irB); tn1, out1, sz1 = native(optB)
        res[p] = {"outputs_equal_vm": a[0] == b[0], "outputs_equal_native": out0 == out1, "output": a[0],
                  "dyn_instr": [a[1], b[1]], "cycles": [a[2], b[2]], "static_size": [static_size(ir), static_size(opt)],
                  "compile_ms_opt_passes": 1000 * tc, "native_secs_gcc_O0": [tn0, tn1], "native_speedup": tn0 / tn1,
                  "binary_bytes": [sz0, sz1], "op_mix_before": a[3], "op_mix_after": b[3]}
        print(p, res[p]["dyn_instr"], round(res[p]["native_speedup"], 2), flush=True)
    return res


def total_cycles(pipeline, cache={}):
    tot = 0
    for p in PROGS:
        ir = compile_to_ir(src(p)); code = ir
        for ps in pipeline: code = PASSES[ps](code)
        key = (p, tuple(code))
        if key not in cache: cache[key] = run_vm(code)[2]
        tot += cache[key]
    return tot


def f2_ablation():
    base = total_cycles([]); res = {"none": base}
    for ps in PASSES: res[f"only {ps}"] = total_cycles([ps] * 2)
    full = sum(run_vm(optimise(compile_to_ir(src(p))))[2] for p in PROGS); res["default pipeline (fixpoint)"] = full
    for ps in PASSES:
        pipe = [x for x in DEFAULT_PIPELINE if x != ps]
        res[f"default minus {ps}"] = sum(run_vm(optimise(compile_to_ir(src(p)), pipe))[2] for p in PROGS)
    return res


def f3_pass_ordering(L=4):
    names = list(PASSES); t0 = time.perf_counter()
    scores = {seq: total_cycles(list(seq)) for seq in itertools.product(names, repeat=L)}
    t_ex = time.perf_counter() - t0
    best_seq = min(scores, key=scores.get); worst = max(scores.values())
    t0 = time.perf_counter(); seq = []; evals = 0
    for _ in range(L):
        cand = {ps: total_cycles(seq + [ps]) for ps in names}; evals += len(cand)
        seq.append(min(cand, key=cand.get))
    t_gr = time.perf_counter() - t0
    import numpy as np
    vals = np.array(list(scores.values()))
    return {"search_space": len(scores), "baseline_no_opt": total_cycles([]),
            "exhaustive_best": {"seq": best_seq, "cycles": scores[best_seq], "secs": t_ex, "evaluations": len(scores)},
            "greedy": {"seq": seq, "cycles": total_cycles(seq), "secs": t_gr, "evaluations": evals},
            "default_order_first4": {"seq": DEFAULT_PIPELINE[:4], "cycles": total_cycles(DEFAULT_PIPELINE[:4])},
            "worst_sequence_cycles": int(worst), "median_sequence_cycles": float(np.median(vals)),
            "num_sequences_achieving_best": int((vals == scores[best_seq]).sum())}


def f4_regalloc():
    res = {}
    for p in PROGS:
        opt = optimise(compile_to_ir(src(p))); ref = run_vm(opt)[0]; nodes, edges, cost = interference(opt)
        res[p] = {"variables": len(nodes), "interference_edges": len(edges)}
        for K in [2, 3, 4, 6]:
            t0 = time.perf_counter(); col, sp = color_chaitin(nodes, edges, cost, K); tch = time.perf_counter() - t0
            t0 = time.perf_counter(); coli, spi, msg = color_ilp(nodes, edges, cost, K); til = time.perf_counter() - t0
            rc = run_vm(rewrite_with_registers(opt, col, sp)); ri = run_vm(rewrite_with_registers(opt, coli, spi))
            assert rc[0] == ref and ri[0] == ref, f"register allocation changed output of {p}"
            # verify colourings are proper
            for a, b in edges:
                assert not (a in col and b in col and col[a] == col[b]); assert not (a in coli and b in coli and coli[a] == coli[b])
            mem = lambda h: h.get("load", 0) + h.get("store", 0)
            res[p][f"K={K}"] = {"chaitin": {"spilled": len(sp), "spill_cost": sum(cost[v] for v in sp), "dyn_mem_ops": mem(rc[3]), "cycles": rc[2], "ms": 1000 * tch},
                                "ilp": {"spilled": len(spi), "spill_cost": sum(cost[v] for v in spi), "dyn_mem_ops": mem(ri[3]), "cycles": ri[2], "ms": 1000 * til, "status": msg[:40]}}
        print(p, {k: (v["chaitin"]["spill_cost"], v["ilp"]["spill_cost"]) for k, v in res[p].items() if k.startswith("K")}, flush=True)
    return res


def f4b_random_graphs(n=30, seeds=10):
    """Harder interference graphs than our small benchmarks produce: random G(n, p) with random
    spill costs. Measures how far the greedy Chaitin-Briggs heuristic is from the ILP optimum."""
    import numpy as np, networkx as nx
    out = {}
    for dens, K in [(0.2, 3), (0.3, 4), (0.4, 5)]:
        gaps, tch, til, opt_ct = [], [], [], 0
        for sd in range(seeds):
            G = nx.gnp_random_graph(n, dens, seed=sd); rng = np.random.default_rng(sd)
            nodes = [f"v{i}" for i in G.nodes]; edges = {(f"v{a}", f"v{b}") for a, b in G.edges}
            cost = {v: float(rng.choice([1, 10, 100])) for v in nodes}
            t0 = time.perf_counter(); c1, s1 = color_chaitin(nodes, edges, cost, K); tch.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); c2, s2, msg = color_ilp(nodes, edges, cost, K, time_limit=60); til.append(time.perf_counter() - t0)
            a, b = sum(cost[v] for v in s1), sum(cost[v] for v in s2); opt_ct += "Optimal" in msg
            gaps.append((a - b) / b * 100 if b > 0 else 0.0)
        out[f"n={n},p={dens},K={K}"] = {"chaitin_excess_spill_cost_%_mean": float(np.mean(gaps)), "max_%": float(np.max(gaps)),
                                         "instances_where_chaitin_suboptimal": int(sum(g > 1e-9 for g in gaps)),
                                         "chaitin_ms": 1000 * float(np.mean(tch)), "ilp_ms": 1000 * float(np.mean(til)), "ilp_proved_optimal": opt_ct}
    return out


def plot(f1):
    p = plt(); fig, axes = p.subplots(1, 2, figsize=(11, 4)); import numpy as np
    x = np.arange(len(PROGS)); w = 0.38
    axes[0].bar(x - w / 2, [f1[q]["dyn_instr"][0] for q in PROGS], w, label="unoptimised")
    axes[0].bar(x + w / 2, [f1[q]["dyn_instr"][1] for q in PROGS], w, label="optimised")
    axes[0].set_xticks(x); axes[0].set_xticklabels(PROGS, rotation=20, fontsize=8); axes[0].set_ylabel("executed IR instructions"); axes[0].legend()
    axes[1].bar(x, [f1[q]["native_speedup"] for q in PROGS]); axes[1].axhline(1, color="k", lw=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(PROGS, rotation=20, fontsize=8); axes[1].set_ylabel("native speedup (gcc -O0 on emitted C)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir(T), "F_speedup.png")); p.close(fig)


if __name__ == "__main__" and "--only-f4b" in sys.argv:
    r = json.load(open(os.path.join(outdir(T), "compiler.json"))); r["F4b_regalloc_random_graphs"] = f4b_random_graphs()
    print(json.dumps(r["F4b_regalloc_random_graphs"], indent=1)); save_json(T, "compiler.json", r); sys.exit()

if __name__ == "__main__":
    r = {"F1_baseline_vs_optimised": f1_baseline_vs_opt()}; plot(r["F1_baseline_vs_optimised"])
    r["F2_ablation_total_cycles"] = f2_ablation(); print("F2", r["F2_ablation_total_cycles"], flush=True)
    r["F4_register_allocation"] = f4_regalloc()
    r["F4b_regalloc_random_graphs"] = f4b_random_graphs(); print("F4b", r["F4b_regalloc_random_graphs"], flush=True)
    r["F3_pass_ordering"] = f3_pass_ordering(); print("F3", r["F3_pass_ordering"], flush=True)
    r["F5_fuzzing"] = fuzz(1000); print("F5", r["F5_fuzzing"], flush=True)
    save_json(T, "compiler.json", r)
