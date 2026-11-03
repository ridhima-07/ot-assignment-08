"""Differential fuzzing: random MiniC programs must print the same values before and after
every optimisation pipeline (and after register allocation). Returns counts of mismatches."""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trackF.minic import compile_to_ir, run_vm
from trackF.passes import optimise, PASSES, interference, color_chaitin, rewrite_with_registers

VARS = ["a", "b", "c", "d", "e"]


def rexpr(r, depth=0):
    if depth > 2 or r.random() < 0.3:
        return str(r.randint(0, 9)) if r.random() < 0.4 else r.choice(VARS)
    op = r.choice(["+", "-", "*", "<", "==", "!=", "%", "/"])
    lhs, rhs = rexpr(r, depth + 1), rexpr(r, depth + 1)
    if op in "%/": rhs = str(r.randint(1, 9))                    # avoid division by zero
    return f"({lhs} {op} {rhs})"


def rstmt(r, depth, loopvar):
    k = r.random()
    if depth < 2 and k < 0.15:
        lv = f"i{depth}"
        body = " ".join(rstmt(r, depth + 1, lv) for _ in range(r.randint(1, 4)))
        return f"{lv} = 0; while ({lv} < {r.randint(1, 6)}) {{ {body} {lv} = {lv} + 1; }}"
    if depth < 3 and k < 0.3:
        th = " ".join(rstmt(r, depth + 1, loopvar) for _ in range(r.randint(1, 3)))
        el = " ".join(rstmt(r, depth + 1, loopvar) for _ in range(r.randint(0, 2)))
        return f"if ({rexpr(r)}) {{ {th} }}" + (f" else {{ {el} }}" if el else "")
    if k < 0.4: return f"print {rexpr(r)};"
    return f"{r.choice(VARS)} = {rexpr(r)};"


def rprog(seed):
    r = random.Random(seed)
    init = " ".join(f"{v} = {r.randint(-5, 9)};" for v in VARS)
    return init + " " + " ".join(rstmt(r, 0, None) for _ in range(r.randint(3, 10))) + " print a + b + c + d + e;"


def fuzz(n=1000, K=3):
    bad = []; rng = random.Random(0)
    for s in range(n):
        src = rprog(s); ir = compile_to_ir(src); ref = run_vm(ir)[0]
        pipes = [list(PASSES)] + [[rng.choice(list(PASSES)) for _ in range(4)] for _ in range(2)]
        for p in pipes:
            opt = optimise(ir, p)
            if run_vm(opt)[0] != ref: bad.append((s, "+".join(p))); break
            nodes, edges, cost = interference(opt); col, sp = color_chaitin(nodes, edges, cost, K)
            if run_vm(rewrite_with_registers(opt, col, sp))[0] != ref: bad.append((s, "regalloc after " + "+".join(p))); break
    return {"programs": n, "pipelines_per_program": 3, "mismatches": len(bad), "examples": bad[:10]}


if __name__ == "__main__":
    print(fuzz(int(sys.argv[1]) if len(sys.argv) > 1 else 300))
