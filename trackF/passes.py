"""Optimisation passes over MiniC three-address IR.

  constprop  global constant propagation + folding (iterative dataflow over the CFG),
             including folding of constant branches
  copyprop   local copy propagation
  cse        local common-subexpression elimination (value numbering per basic block)
  dce        dead-code elimination (liveness) + unreachable-block removal
  licm       loop-invariant code motion (dominators -> natural loops -> preheader)
  sr         strength reduction / algebraic simplification (x*2^k -> x<<k, x*1, x+0, ...)
  regalloc   graph-colouring (Chaitin-Briggs) or ILP register allocation with spill code
"""
from collections import defaultdict
import numpy as np

PURE_DEF = ("const", "mov", "bin")
COMMUT = {"+", "*", "==", "!="}


def uses(ins):
    k = ins[0]
    if k == "mov": c = [ins[2]]
    elif k == "bin": c = [ins[3], ins[4]]
    elif k in ("print", "jz"): c = [ins[1]]
    elif k == "store": c = [ins[2]]
    else: c = []
    return [x for x in c if isinstance(x, str)]


def defs(ins):
    if ins[0] in ("const", "mov", "load"): return [ins[1]]
    if ins[0] == "bin": return [ins[2]]
    return []


def replace_uses(ins, f):
    k = ins[0]
    if k == "mov": return ("mov", ins[1], f(ins[2]))
    if k == "bin": return ("bin", ins[1], ins[2], f(ins[3]), f(ins[4]))
    if k == "print": return ("print", f(ins[1]))
    if k == "jz": return ("jz", f(ins[1]), ins[2])
    if k == "store": return ("store", ins[1], f(ins[2]))
    return ins


def may_trap(ins):
    return ins[0] == "bin" and ins[1] in ("/", "%") and not (isinstance(ins[4], int) and ins[4] != 0)


# ---------------------------------------------------------------- CFG
class CFG:
    def __init__(self, code):
        self.code = code; leaders = {0}
        for i, ins in enumerate(code):
            if ins[0] == "label": leaders.add(i)
            if ins[0] in ("jmp", "jz") and i + 1 < len(code): leaders.add(i + 1)
        starts = sorted(l for l in leaders if l < len(code))
        self.blocks = [list(range(s, e)) for s, e in zip(starts, starts[1:] + [len(code)])]
        lab = {code[b[0]][1]: k for k, b in enumerate(self.blocks) if b and code[b[0]][0] == "label"}
        self.succ = defaultdict(list); self.pred = defaultdict(list)
        for k, b in enumerate(self.blocks):
            last = code[b[-1]] if b else None
            if last and last[0] == "jmp": ss = [lab[last[1]]]
            elif last and last[0] == "jz": ss = [lab[last[2]]] + ([k + 1] if k + 1 < len(self.blocks) else [])
            else: ss = [k + 1] if k + 1 < len(self.blocks) else []
            for s in ss:
                if s not in self.succ[k]: self.succ[k].append(s); self.pred[s].append(k)
        self.label_block = lab

    def reachable(self):
        seen, st = {0}, [0]
        while st:
            for s in self.succ[st.pop()]:
                if s not in seen: seen.add(s); st.append(s)
        return seen

    def dominators(self):
        n = len(self.blocks); allb = set(range(n)); dom = {b: set(allb) for b in range(n)}; dom[0] = {0}
        reach = self.reachable(); changed = True
        while changed:
            changed = False
            for b in range(1, n):
                if b not in reach: continue
                ps = [dom[p] for p in self.pred[b] if p in reach]
                new = ({b} | set.intersection(*ps)) if ps else {b}
                if new != dom[b]: dom[b] = new; changed = True
        return dom

    def natural_loops(self):
        dom = self.dominators(); loops = []
        for n in self.reachable():
            for h in self.succ[n]:
                if h in dom[n]:                               # back edge n -> h
                    body, st = {h, n}, [n]
                    while st:
                        m = st.pop()
                        for p in self.pred[m]:
                            if p not in body: body.add(p); st.append(p)
                    loops.append((h, body))
        return loops

    def liveness(self):
        n = len(self.blocks); use_b, def_b = [], []
        for b in self.blocks:
            u, d = set(), set()
            for i in b:
                for x in uses(self.code[i]):
                    if x not in d: u.add(x)
                d.update(defs(self.code[i]))
            use_b.append(u); def_b.append(d)
        IN = [set() for _ in range(n)]; OUT = [set() for _ in range(n)]; changed = True
        while changed:
            changed = False
            for k in reversed(range(n)):
                out = set().union(*[IN[s] for s in self.succ[k]]) if self.succ[k] else set()
                inn = use_b[k] | (out - def_b[k])
                if out != OUT[k] or inn != IN[k]: OUT[k], IN[k] = out, inn; changed = True
        return IN, OUT


# ---------------------------------------------------------------- passes
NAC = object()


def constprop(code):
    cfg = CFG(code); allv = {x for ins in code for x in uses(ins) + defs(ins)}
    def meet(states):
        out = {}
        for st in states:
            for v, c in st.items():
                if v not in out: out[v] = c
                elif out[v] is not NAC and (c is NAC or c != out[v]): out[v] = NAC
        return out
    def transfer(state, ins):
        st = state; val = lambda x: x if isinstance(x, int) else st.get(x, NAC)
        k = ins[0]
        if k == "const": st[ins[1]] = ins[2]
        elif k == "mov": st[ins[1]] = val(ins[2])
        elif k == "bin":
            a, b = val(ins[3]), val(ins[4])
            if a is NAC or b is NAC: st[ins[2]] = NAC
            else:
                try:
                    from trackF.minic import binop
                    st[ins[2]] = binop(ins[1], a, b)
                except ZeroDivisionError: st[ins[2]] = NAC
        elif k == "load": st[ins[1]] = NAC
        return st
    n = len(cfg.blocks); OUT = [None] * n; IN = [None] * n
    reach = cfg.reachable(); changed = True
    while changed:
        changed = False
        for k in range(n):
            if k not in reach: continue
            preds = [OUT[p] for p in cfg.pred[k] if OUT[p] is not None]
            inn = meet(preds + ([{v: NAC for v in allv}] if k == 0 else []))
            st = dict(inn)
            for i in cfg.blocks[k]: st = transfer(st, code[i])
            IN[k] = inn
            if st != OUT[k]: OUT[k] = st; changed = True
    new = list(code)
    for k in range(n):
        if IN[k] is None: continue
        st = dict(IN[k])
        for i in cfg.blocks[k]:
            ins = code[i]
            sub = lambda x, st=st: (st[x] if isinstance(x, str) and x in st and st[x] is not NAC else x)
            r = replace_uses(ins, sub)
            if r[0] == "bin" and isinstance(r[3], int) and isinstance(r[4], int):
                try:
                    from trackF.minic import binop
                    r = ("const", r[2], binop(r[1], r[3], r[4]))
                except ZeroDivisionError: pass
            if r[0] == "mov" and isinstance(r[2], int): r = ("const", r[1], r[2])
            if r[0] == "jz" and isinstance(r[1], int):
                r = ("jmp", r[2]) if r[1] == 0 else ("nop",)
            new[i] = r; st = transfer(st, ins)
    return [i for i in new if i[0] != "nop"]


def copyprop(code):
    cfg = CFG(code); new = list(code)
    for b in cfg.blocks:
        copies = {}
        for i in b:
            ins = replace_uses(code[i], lambda x: copies.get(x, x)); new[i] = ins
            for d in defs(ins):
                copies = {k: v for k, v in copies.items() if k != d and v != d}
            if ins[0] == "mov" and isinstance(ins[2], str) and ins[1] != ins[2]: copies[ins[1]] = ins[2]
            if ins[0] == "mov" and isinstance(ins[2], int): copies[ins[1]] = ins[2]
    return new


def cse(code):
    cfg = CFG(code); new = list(code)
    for b in cfg.blocks:
        table = {}
        for i in b:
            ins = code[i]
            if ins[0] == "bin":
                a, c = ins[3], ins[4]
                key = (ins[1],) + (tuple(sorted([a, c], key=str)) if ins[1] in COMMUT else (a, c))
                if key in table: new[i] = ("mov", ins[2], table[key])
                d = ins[2]
                table = {k: v for k, v in table.items() if d not in k[1:] and v != d}
                if key not in table and d not in key[1:]: table[key] = d
            else:
                for d in defs(ins): table = {k: v for k, v in table.items() if d not in k[1:] and v != d}
    return new


def dce(code):
    changed = True
    while changed:
        changed = False; cfg = CFG(code); reach = cfg.reachable()
        keep = set(i for k in reach for i in cfg.blocks[k])
        code2 = [code[i] for i in range(len(code)) if i in keep]
        if len(code2) != len(code): code, changed = code2, True; continue
        IN, OUT = cfg.liveness(); dead = set()
        for k, b in enumerate(cfg.blocks):
            live = set(OUT[k])
            for i in reversed(b):
                ins = code[i]; ds = defs(ins)
                if ins[0] in PURE_DEF and ds and ds[0] not in live and not may_trap(ins): dead.add(i); continue
                if ins[0] == "mov" and ins[1] == ins[2]: dead.add(i); continue
                live -= set(ds); live |= set(uses(ins))
        # jump to the very next label
        for i, ins in enumerate(code):
            if ins[0] == "jmp" and i + 1 < len(code) and code[i + 1] == ("label", ins[1]): dead.add(i)
        if dead: code = [c for i, c in enumerate(code) if i not in dead]; changed = True
    return code


def licm(code):
    cfg = CFG(code); loops = sorted(cfg.natural_loops(), key=lambda l: len(l[1]))
    defcount = defaultdict(int)
    for ins in code:
        for d in defs(ins): defcount[d] += 1
    moved = {}; hoist_before = defaultdict(list)
    for h, body in loops:
        head_idx = cfg.blocks[h][0]
        if code[head_idx][0] != "label": continue
        hl = code[head_idx][1]
        # header must be entered from outside the loop only by fall-through (true for structured code)
        outside_jumps = [i for k in range(len(cfg.blocks)) if k not in body for i in cfg.blocks[k]
                         if code[i][0] in ("jmp", "jz") and code[i][-1] == hl]
        if outside_jumps: continue
        idxs = sorted(i for k in body for i in cfg.blocks[k] if i not in moved)
        loop_defs = defaultdict(int)
        for i in idxs:
            for d in defs(code[i]): loop_defs[d] += 1
        changed = True
        while changed:
            changed = False
            for i in idxs:
                if i in moved: continue
                ins = code[i]
                if ins[0] not in ("bin", "const") or may_trap(ins): continue
                d = defs(ins)[0]
                if not (d.startswith("t") and d[1:].isdigit() and defcount[d] == 1): continue
                if all(isinstance(x, int) or loop_defs.get(x, 0) == 0 for x in uses(ins)):
                    moved[i] = head_idx; hoist_before[head_idx].append(ins); loop_defs[d] -= 1; changed = True
    if not moved: return code
    out = []
    for i, ins in enumerate(code):
        if i in hoist_before: out.extend(hoist_before[i])
        if i not in moved: out.append(ins)
    return out


def sr(code):
    out = []
    for ins in code:
        if ins[0] == "bin":
            op, d, a, b = ins[1:]
            if op == "*" and isinstance(a, int) and not isinstance(b, int): a, b = b, a
            if op == "*" and isinstance(b, int):
                if b == 0: out.append(("const", d, 0)); continue
                if b == 1: out.append(("mov", d, a)); continue
                if b > 1 and b & (b - 1) == 0: out.append(("bin", "shl", d, a, b.bit_length() - 1)); continue
            if op in ("+", "-") and b == 0: out.append(("mov", d, a)); continue
            if op == "+" and a == 0: out.append(("mov", d, b)); continue
            if op == "/" and b == 1: out.append(("mov", d, a)); continue
        out.append(ins)
    return out


PASSES = {"constprop": constprop, "copyprop": copyprop, "cse": cse, "dce": dce, "licm": licm, "sr": sr}
DEFAULT_PIPELINE = ["constprop", "copyprop", "cse", "sr", "licm", "copyprop", "constprop", "dce"]


def optimise(code, pipeline=DEFAULT_PIPELINE, fixpoint=True):
    for _ in range(5 if fixpoint else 1):
        before = code
        for p in pipeline: code = PASSES[p](code)
        if code == before: break
    return code


# ---------------------------------------------------------------- register allocation
def interference(code):
    cfg = CFG(code); IN, OUT = cfg.liveness(); edges = set(); nodes = set(); depth = defaultdict(int)
    for h, body in cfg.natural_loops():
        for k in body:
            for i in cfg.blocks[k]: depth[i] += 1
    cost = defaultdict(float)
    for k, b in enumerate(cfg.blocks):
        live = set(OUT[k])
        for i in reversed(b):
            ins = code[i]; ds = defs(ins); us = uses(ins)
            for v in ds + us: nodes.add(v); cost[v] += 10 ** depth[i]
            for d in ds:
                for v in live:
                    if v != d and not (ins[0] == "mov" and v == ins[2]): edges.add(tuple(sorted((d, v))))
            live -= set(ds); live |= set(us)
    return sorted(nodes), edges, cost


def color_chaitin(nodes, edges, cost, K):
    adj = {v: set() for v in nodes}
    for a, b in edges: adj[a].add(b); adj[b].add(a)
    g = {v: set(adj[v]) for v in nodes}; stack = []
    while g:
        low = [v for v in g if len(g[v]) < K]
        v = low[0] if low else min(g, key=lambda u: cost[u] / max(len(g[u]), 1))   # potential spill: cheap & high degree
        stack.append(v)
        for u in g[v]: g[u].discard(v)
        del g[v]
    color, spilled = {}, set()
    while stack:                                                                    # optimistic colouring (Briggs)
        v = stack.pop(); used = {color[u] for u in adj[v] if u in color}
        free = [c for c in range(K) if c not in used]
        if free: color[v] = free[0]
        else: spilled.add(v)
    return color, spilled


def color_ilp(nodes, edges, cost, K, time_limit=30):
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix
    n = len(nodes); ix = {v: i for i, v in enumerate(nodes)}; nv = n * K + n
    c = np.zeros(nv); c[n * K:] = [cost[v] for v in nodes]
    rows = len(nodes) + len(edges) * K; A = lil_matrix((rows, nv)); lo = np.zeros(rows); hi = np.zeros(rows); r = 0
    for v in nodes:
        for k in range(K): A[r, ix[v] * K + k] = 1
        A[r, n * K + ix[v]] = 1; lo[r] = hi[r] = 1; r += 1
    for a, b in edges:
        for k in range(K): A[r, ix[a] * K + k] = 1; A[r, ix[b] * K + k] = 1; lo[r] = 0; hi[r] = 1; r += 1
    # symmetry breaking: first node can only take colour 0
    res = milp(c, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=np.ones(nv), bounds=Bounds(0, 1),
               options={"time_limit": time_limit})
    x = np.round(res.x).astype(int); color, spilled = {}, set()
    for v in nodes:
        if x[n * K + ix[v]]: spilled.add(v)
        else: color[v] = int(np.argmax(x[ix[v] * K:(ix[v] + 1) * K]))
    return color, spilled, res.message


def rewrite_with_registers(code, color, spilled):
    slot = {v: f"m_{v}" for v in spilled}; out = []
    for ins in code:
        scratch = iter(["S0", "S1"]); loaded = {}
        for u in uses(ins):
            if u in spilled and u not in loaded:
                r = next(scratch); loaded[u] = r; out.append(("load", r, slot[u]))
        mp = lambda x: loaded.get(x, f"R{color[x]}" if isinstance(x, str) and x in color else x)
        ins2 = replace_uses(ins, mp); ds = defs(ins)
        if ds:
            d = ds[0]; nd = "S0" if d in spilled else f"R{color[d]}"
            k = ins2[0]
            ins2 = (k, nd, *ins2[2:]) if k in ("const", "mov", "load") else ("bin", ins2[1], nd, ins2[3], ins2[4])
            out.append(ins2)
            if d in spilled: out.append(("store", slot[d], "S0"))
        else:
            out.append(ins2)
    return out
