"""Loop-invariant code motion for the MiniC three-address IR (see minic.py).

licm(code) returns a new instruction list in which loop-invariant, pure computations
of single-definition temporaries (t1, t2, ...) are moved into a freshly created loop
preheader (a label + straight-line code placed immediately before the loop header).

Hoisting rules (all must hold):
  * the destination is a temporary named t<digits> with exactly one definition in the program;
  * the instruction is const, mov or bin (never print, jumps, labels or variable writes);
  * "/" and "%" are hoisted only when the divisor is a known nonzero constant (an int literal,
    or a single-definition temp holding a nonzero const); "shl" only with a constant in [0, 64);
  * every operand is an int literal, a name not written anywhere inside the loop, or a temp
    that has already been hoisted from this loop.
"""
import itertools
import re
from collections import Counter

TEMP = re.compile(r"t\d+")
JUMPS = ("jmp", "jz")


def _dst(ins):
    k = ins[0]
    if k == "bin":
        return ins[2]
    if k in ("const", "mov"):
        return ins[1]
    return None


def _srcs(ins):
    k = ins[0]
    if k == "bin":
        return (ins[3], ins[4])
    if k == "mov":
        return (ins[2],)
    return ()


def _split_blocks(code):
    """Basic blocks: {"label": name-or-None, "ins": [non-label instructions]}."""
    blocks, cur = [], None
    for ins in code:
        if ins[0] == "label":
            if cur is not None:
                blocks.append(cur)
            cur = {"label": ins[1], "ins": []}
        else:
            if cur is None:
                cur = {"label": None, "ins": []}
            cur["ins"].append(ins)
            if ins[0] in JUMPS:
                blocks.append(cur)
                cur = None
    if cur is not None:
        blocks.append(cur)
    return blocks


def _successors(blocks):
    index_of = {b["label"]: i for i, b in enumerate(blocks) if b["label"] is not None}
    n, succ = len(blocks), []
    for i, b in enumerate(blocks):
        last = b["ins"][-1] if b["ins"] else None
        fall = [i + 1] if i + 1 < n else []
        if last is not None and last[0] == "jmp":
            succ.append([index_of[last[1]]])
        elif last is not None and last[0] == "jz":
            succ.append([index_of[last[2]]] + fall)
        else:
            succ.append(fall)
    return succ


def _dominators(succ):
    n = len(succ)
    preds = [[] for _ in range(n)]
    for i, ss in enumerate(succ):
        for s in ss:
            preds[s].append(i)
    reach, stack = set(), ([0] if n else [])
    while stack:
        x = stack.pop()
        if x not in reach:
            reach.add(x)
            stack.extend(succ[x])
    dom = [set(reach) for _ in range(n)]
    if n:
        dom[0] = {0}
    changed = True
    while changed:
        changed = False
        for b in range(1, n):
            if b not in reach:
                continue
            ps = [dom[p] for p in preds[b] if p in reach]
            new = {b} | (set.intersection(*ps) if ps else set())
            if new != dom[b]:
                dom[b], changed = new, True
    return dom, preds, reach


def _natural_loops(succ, dom, preds, reach):
    """header block index -> set of block indices in the natural loop (back edges merged per header)."""
    loops = {}
    for b in sorted(reach):
        for h in succ[b]:
            if h in dom[b]:  # back edge b -> h
                body = loops.setdefault(h, {h})
                stack = [b]
                while stack:
                    x = stack.pop()
                    if x not in body:
                        body.add(x)
                        stack.extend(p for p in preds[x] if p in reach)
    return loops


def _hoist_candidates(blocks, body, def_count, const_val):
    """Return (keys, instructions) of hoistable instructions in loop `body`, in a dependency-safe order."""
    loop_defs = set()
    for bi in body:
        for ins in blocks[bi]["ins"]:
            d = _dst(ins)
            if d is not None:
                loop_defs.add(d)
    hoisted = set()

    def const_of(x):
        return x if isinstance(x, int) else const_val.get(x)

    def invariant(x):
        return isinstance(x, int) or x in hoisted or x not in loop_defs

    def ok(ins):
        d = _dst(ins)
        if d is None or not TEMP.fullmatch(d) or def_count[d] != 1:
            return False
        if ins[0] == "bin":
            op, c = ins[1], const_of(ins[4])
            if op in ("/", "%") and (c is None or c == 0):
                return False
            if op == "shl" and (c is None or not 0 <= c < 64):
                return False
        return all(invariant(s) for s in _srcs(ins))

    cands = [(bi, ii) for bi in sorted(body) for ii in range(len(blocks[bi]["ins"]))]
    taken, order, changed = set(), [], True
    while changed:
        changed = False
        for key in cands:
            if key in taken:
                continue
            ins = blocks[key[0]]["ins"][key[1]]
            if ok(ins):
                taken.add(key)
                order.append(ins)
                hoisted.add(_dst(ins))
                changed = True
    return taken, order


def _emit(blocks):
    out = []
    for blk in blocks:
        if blk["label"] is not None:
            out.append(("label", blk["label"]))
        out.extend(blk["ins"])
    return out


def licm(code):
    """Return a new IR list with loop-invariant temporaries hoisted into loop preheaders."""
    code = list(code)
    while True:
        def_count = Counter(d for d in (_dst(i) for i in code) if d is not None)
        const_val = {i[1]: i[2] for i in code
                     if i[0] == "const" and TEMP.fullmatch(i[1]) and def_count[i[1]] == 1}
        blocks = _split_blocks(code)
        succ = _successors(blocks)
        dom, preds, reach = _dominators(succ)
        loops = _natural_loops(succ, dom, preds, reach)

        for h in sorted(loops, key=lambda k: len(loops[k])):  # innermost (smallest) first
            body = loops[h]
            taken, order = _hoist_candidates(blocks, body, def_count, const_val)
            if not order:
                continue
            labels = {i[1] for i in code if i[0] == "label"}
            pre = next(f"Lph{n}" for n in itertools.count(1) if f"Lph{n}" not in labels)
            header_label = blocks[h]["label"]
            # remove hoisted instructions from loop blocks (terminators are never hoisted)
            for bi in body:
                blocks[bi]["ins"] = [x for ii, x in enumerate(blocks[bi]["ins"]) if (bi, ii) not in taken]
            # outside predecessors that jump to the header now jump to the preheader
            for bi, blk in enumerate(blocks):
                if bi in body or not blk["ins"]:
                    continue
                last = blk["ins"][-1]
                if last[0] == "jmp" and last[1] == header_label:
                    blk["ins"][-1] = ("jmp", pre)
                elif last[0] == "jz" and last[2] == header_label:
                    blk["ins"][-1] = ("jz", last[1], pre)
            # preheader falls through into the header; fall-through entries now reach it too
            blocks.insert(h, {"label": pre, "ins": order})
            code = _emit(blocks)
            break
        else:
            return code
