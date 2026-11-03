"""Track J: test-suite minimisation and prioritisation on more-itertools (public OSS project).

PROFILE   run the full suite with per-test coverage (coverage.py dynamic contexts) and per-test time.
PROBLEM   x_t in {0,1} for each test t. minimise sum_t time_t x_t
          s.t. every source line covered by the full suite stays covered: sum_{t covers l} x_t >= 1.
          (weighted set cover; NP-hard)
METHODS   full suite (baseline) | greedy (max new lines per second) | ILP (HiGHS, exact) | random subset
VALIDATE  (1) re-run each reduced suite for real and time it; (2) mutation testing: AST mutants of the
          library, kill matrix from the full suite, mutation score of every subset;
          (3) prioritisation: APFD of original / random / greedy-additional-coverage orders.
"""
import sys, os, json, ast, random, shutil, subprocess, tempfile, time, re
import xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
from common.util import save_json, outdir, plt

T = "trackJ"
REPO = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/ext/more-itertools"
WORK = os.path.join(tempfile.gettempdir(), "a8_trackJ")
PY = sys.executable


def sh(cmd, cwd, env=None, timeout=1800):
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)


def profile():
    """Per-test time (junit xml) and per-test covered lines (coverage dynamic contexts)."""
    rc = os.path.join(WORK, "covrc"); open(rc, "w").write("[run]\nsource = more_itertools\ndynamic_context = test_function\n")
    jx = os.path.join(WORK, "junit.xml")
    t0 = time.perf_counter()
    sh([PY, "-m", "coverage", "run", f"--rcfile={rc}", "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={jx}", "tests"], REPO)
    t_cov = time.perf_counter() - t0
    sh([PY, "-m", "coverage", "json", f"--rcfile={rc}", "--show-contexts", "-o", os.path.join(WORK, "cov.json")], REPO)
    cov = json.load(open(os.path.join(WORK, "cov.json")))
    test_lines = {}
    for fname, f in cov["files"].items():
        for line, ctxs in f.get("contexts", {}).items():
            for c in ctxs:
                if not c: continue
                tid = c.split("|")[0]                                           # tests.test_more.SomeTest.test_x
                test_lines.setdefault(tid, set()).add(f"{fname}:{line}")
    times = {}
    for tc in ET.parse(jx).getroot().iter("testcase"):
        tid = f"{tc.get('classname')}.{tc.get('name')}"; times[tid] = float(tc.get("time", 0))
    # Coverage names a context by the class that DEFINES the test method (e.g. a mixin), pytest/junit by the
    # subclass that RUNS it. Map such contexts onto every junit test in the same module with the same method name.
    # (Bug fixed: without this, lines covered only by inherited tests were uncoverable and the ILP was infeasible.)
    for ctx in [c for c in list(test_lines) if c not in times]:
        mod, meth = ctx.rsplit(".", 2)[0], ctx.rsplit(".", 1)[1]
        for jid in [j for j in times if j.startswith(mod + ".") and j.endswith("." + meth) and j not in test_lines]:
            test_lines[jid] = set(test_lines[ctx])
        test_lines.pop(ctx)
    tests = sorted(times)
    lines = sorted(set().union(*[test_lines.get(t, set()) for t in tests]))
    return tests, times, test_lines, lines, t_cov, cov["totals"]


def to_nodeid(tid):
    """tests.test_more.ChunkedTests.test_even -> tests/test_more.py::ChunkedTests::test_even"""
    parts = tid.split(".")
    for k in range(len(parts), 0, -1):
        path = os.path.join(*parts[:k]) + ".py"
        if os.path.exists(os.path.join(REPO, path)): return path + "::" + "::".join(parts[k:])
    return tid


def greedy_cover(tests, times, test_lines, universe):
    unc = set(universe); chosen = []
    while unc:
        best = max(tests, key=lambda t: len(test_lines.get(t, set()) & unc) / max(times[t], 1e-4))
        if not test_lines.get(best, set()) & unc: break
        chosen.append(best); unc -= test_lines[best]
    return chosen


def ilp_cover(tests, times, test_lines, universe):
    li = {l: k for k, l in enumerate(universe)}; A = lil_matrix((len(universe), len(tests)))
    for j, t in enumerate(tests):
        for l in test_lines.get(t, ()): A[li[l], j] = 1
    c = np.array([max(times[t], 1e-4) for t in tests])
    t0 = time.perf_counter()
    r = milp(c, constraints=LinearConstraint(A.tocsr(), 1, np.inf), integrality=np.ones(len(tests)), bounds=Bounds(0, 1),
             options={"time_limit": 300})
    return [t for j, t in enumerate(tests) if r.x[j] > 0.5], time.perf_counter() - t0, r.message


def run_subset(tests, repo=REPO, reps=3):
    ids = [to_nodeid(t) for t in tests] or ["tests"]          # empty list = the whole suite
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = sh([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-x", *ids], repo); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), r.returncode, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""


# ---------------- mutation testing ----------------
class Mutator(ast.NodeTransformer):
    SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
            ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Mult: ast.FloorDiv}

    def __init__(self, target): self.k, self.target, self.done = 0, target, None

    def _hit(self, node, desc):
        self.k += 1
        if self.k - 1 == self.target: self.done = (node.lineno, desc); return True
        return False

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if type(node.op) in self.SWAP and self._hit(node, f"{type(node.op).__name__}->{self.SWAP[type(node.op)].__name__}"):
            node.op = self.SWAP[type(node.op)]()
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if type(node.ops[0]) in self.SWAP and self._hit(node, f"{type(node.ops[0]).__name__}->{self.SWAP[type(node.ops[0])].__name__}"):
            node.ops[0] = self.SWAP[type(node.ops[0])]()
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, bool) and self._hit(node, f"{node.value}->{not node.value}"): return ast.copy_location(ast.Constant(not node.value), node)
        if type(node.value) is int and node.value in (0, 1) and self._hit(node, f"{node.value}->{1 - node.value}"):
            return ast.copy_location(ast.Constant(1 - node.value), node)
        return node


def mutants(n, covered_lines, seed=0):
    rng = random.Random(seed); out = []
    for mod in ["more.py", "recipes.py"]:
        src = open(os.path.join(REPO, "more_itertools", mod)).read(); total = Mutator(-1); total.visit(ast.parse(src))
        sites = list(range(total.k)); rng.shuffle(sites)
        for s in sites:
            m = Mutator(s); tree = m.visit(ast.parse(src))
            if m.done and f"more_itertools/{mod}:{m.done[0]}" in covered_lines:
                out.append((mod, s, m.done, ast.unparse(tree)))
            if len([o for o in out if o[0] == mod]) >= n // 2: break
    return out


def kill_matrix(muts, tests):
    mdir = os.path.join(WORK, "mutrepo"); kills = []
    if os.path.exists(mdir): shutil.rmtree(mdir)
    shutil.copytree(REPO, mdir, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    for k, (mod, s, desc, code) in enumerate(muts):
        path = os.path.join(mdir, "more_itertools", mod); orig = open(path).read(); open(path, "w").write(code)
        jx = os.path.join(WORK, f"m{k}.xml")
        try:
            sh([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={jx}", "tests"], mdir, timeout=300)
            failed = set()
            for tc in ET.parse(jx).getroot().iter("testcase"):
                if tc.find("failure") is not None or tc.find("error") is not None: failed.add(f"{tc.get('classname')}.{tc.get('name')}")
        except Exception:
            failed = {"__timeout__"}
        kills.append(failed); open(path, "w").write(orig)
        print(f"mutant {k} {mod}:{desc} killed_by={len(failed)}", flush=True)
    return kills


def apfd(order, kills):
    """Average percentage of faults detected for a test order (only mutants killed by the full suite)."""
    pos = {t: i + 1 for i, t in enumerate(order)}; n = len(order); m = [k for k in kills if k and "__timeout__" not in k]
    tf = [min(pos.get(t, n + 1) for t in k) for k in m]
    return 1 - sum(tf) / (n * len(m)) + 1 / (2 * n)


def main(n_mutants=40):
    os.makedirs(WORK, exist_ok=True)
    tests, times, test_lines, lines, t_cov, totals = profile()
    full_time = sum(times.values())
    prof = sorted(times.items(), key=lambda kv: -kv[1])[:10]
    out = {"repo_commit": sh(["git", "rev-parse", "HEAD"], REPO).stdout.strip(), "tests": len(tests), "covered_lines": len(lines),
           "coverage_percent": totals["percent_covered"], "sum_test_time_s": full_time,
           "bottleneck_top10_tests": [{"test": t, "secs": s, "share_%": 100 * s / full_time} for t, s in prof]}
    t0 = time.perf_counter(); g = greedy_cover(tests, times, test_lines, lines); tg = time.perf_counter() - t0
    il, ti, msg = ilp_cover(tests, times, test_lines, lines)
    rng = random.Random(0); rnd = rng.sample(tests, len(il))
    sets = {"full suite": tests, "greedy": g, "ILP": il, "random (same size as ILP)": rnd}
    for name, S in sets.items():
        cov = set().union(*[test_lines.get(t, set()) for t in S])
        wall, rc, tail = run_subset(S) if name != "full suite" else run_subset([], reps=3)
        out[name] = {"n_tests": len(S), "sum_test_time_s": sum(times[t] for t in S), "line_coverage_kept_%": 100 * len(cov) / len(lines),
                     "measured_wall_s": wall, "pytest_exit": rc, "pytest_summary": tail}
        print(name, out[name], flush=True)
    out["greedy"]["solve_secs"] = tg; out["ILP"]["solve_secs"] = ti; out["ILP"]["status"] = msg
    # mutation testing
    muts = mutants(n_mutants, set(lines)); kills = kill_matrix(muts, tests)
    killed_full = [k for k in kills if k and "__timeout__" not in k]
    out["mutants"] = {"generated": len(muts), "killed_by_full_suite": len(killed_full),
                      "examples": [f"{m[0]}:{m[2][0]} {m[2][1]}" for m in muts[:8]]}
    for name, S in sets.items():
        Sset = set(S); out[name]["mutation_score_%"] = 100 * sum(bool(k & Sset) for k in killed_full) / max(len(killed_full), 1)
    # prioritisation
    orig = list(tests); rand = tests[:]; random.Random(1).shuffle(rand)
    unc = set(lines); pri = []; rest = set(tests)
    while rest:
        best = max(rest, key=lambda t: (len(test_lines.get(t, set()) & unc) / max(times[t], 1e-4), -times[t]))
        pri.append(best); rest.discard(best); unc -= test_lines.get(best, set())
        if not unc: unc = set(lines)                                           # reset (additional-greedy)
    out["prioritisation_APFD"] = {"original order": apfd(orig, kills), "random order": apfd(rand, kills), "greedy additional coverage/time": apfd(pri, kills)}
    save_json(T, "test_minimization.json", out)
    p = plt(); fig, ax = p.subplots(figsize=(7, 4)); names = list(sets)
    ax.bar(names, [out[n]["measured_wall_s"] for n in names], color=["#888", "#4a7", "#27c", "#c84"])
    for k, n in enumerate(names): ax.text(k, out[n]["measured_wall_s"], f"{out[n]['n_tests']} tests\nmut {out[n]['mutation_score_%']:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("measured suite wall time (s)"); ax.set_title("more-itertools test-suite minimisation"); fig.tight_layout()
    fig.savefig(os.path.join(outdir(T), "J_minimization.png")); p.close(fig)
    print(json.dumps({k: v for k, v in out.items() if k not in ("bottleneck_top10_tests",)}, indent=1, default=str))


if __name__ == "__main__":
    main()
