"""Grade a candidate licm(code) implementation the same way for every AI tool.

usage: python3 docs/tool_comparison/grade.py <path/to/solution.py>
Swaps the candidate into the MiniC pass pipeline, then (1) differential-fuzzes 300 random programs over 3 pipelines
and (2) checks output equality + dynamic instruction counts on the five Track F benchmarks."""
import sys, os, importlib.util, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from trackF import passes, fuzz
from trackF.minic import compile_to_ir, run_vm

spec = importlib.util.spec_from_file_location("cand", sys.argv[1]); cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)
orig = passes.licm


def bench(licm_fn):
    passes.PASSES["licm"] = licm_fn; res = {}
    for p in ["sum_invariant", "cse_poly", "primes", "matmul", "fib_const"]:
        ir = compile_to_ir(open(os.path.join(ROOT, "trackF", "programs", p + ".mc")).read())
        try:
            out = run_vm(passes.optimise(ir)); res[p] = {"output_ok": out[0] == run_vm(ir)[0], "dyn_instr": out[1]}
        except Exception as e:
            res[p] = {"output_ok": False, "error": repr(e)[:120]}
    return res


report = {"reference_licm": bench(orig)}
passes.PASSES["licm"] = cand.licm
t0 = time.perf_counter()
try:
    report["candidate_fuzz"] = fuzz.fuzz(300)
except Exception as e:
    report["candidate_fuzz"] = {"crashed": repr(e)[:200]}
report["fuzz_secs"] = time.perf_counter() - t0
report["candidate_licm"] = bench(cand.licm)
# does the candidate actually hoist anything? (instruction count vs pipeline without licm)
passes.PASSES["licm"] = lambda c: c
report["no_licm"] = bench(lambda c: c)
passes.PASSES["licm"] = orig
print(json.dumps(report, indent=1))
