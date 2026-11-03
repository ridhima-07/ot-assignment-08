"""MiniC: a toy compiler for Track F.

Language
    prog  := stmt*
    stmt  := ID '=' expr ';' | 'print' expr ';'
           | 'while' '(' expr ')' block | 'if' '(' expr ')' block ['else' block]
    block := '{' stmt* '}'
    expr  := comparisons, + - * / %, unary -, parentheses, integers, identifiers (64-bit ints, C semantics)

Pipeline: lexer -> recursive-descent parser (AST) -> three-address IR -> optimisation passes
(see passes.py) -> two back ends:
    * an IR virtual machine that counts executed instructions (and weighted cycles)
    * a C code generator (compiled with gcc -O0 so that gcc does not redo our work) for wall-clock time
"""
import re

# ---------------------------------------------------------------- lexer
TOK = re.compile(r"\s*(?:(\d+)|([A-Za-z_]\w*)|(==|!=|<=|>=|[-+*/%<>=(){};]))")
KEYWORDS = {"while", "if", "else", "print"}


def lex(src):
    src = re.sub(r"//[^\n]*", "", src); pos, out = 0, []
    while pos < len(src):
        if src[pos:].strip() == "": break
        m = TOK.match(src, pos)
        if not m: raise SyntaxError(f"bad char at {pos}: {src[pos:pos + 10]!r}")
        num, ident, op = m.groups(); pos = m.end()
        if num: out.append(("num", int(num)))
        elif ident: out.append(("kw", ident) if ident in KEYWORDS else ("id", ident))
        else: out.append(("op", op))
    out.append(("eof", None)); return out


# ---------------------------------------------------------------- parser -> AST (tuples)
class Parser:
    PREC = [("==", "!="), ("<", ">", "<=", ">="), ("+", "-"), ("*", "/", "%")]

    def __init__(self, toks): self.t, self.i = toks, 0
    def peek(self): return self.t[self.i]
    def eat(self, kind=None, val=None):
        k, v = self.t[self.i]
        if (kind and k != kind) or (val is not None and v != val): raise SyntaxError(f"expected {kind} {val}, got {k} {v}")
        self.i += 1; return v

    def program(self):
        body = []
        while self.peek()[0] != "eof": body.append(self.stmt())
        return ("block", body)

    def block(self):
        self.eat("op", "{"); body = []
        while self.peek() != ("op", "}"): body.append(self.stmt())
        self.eat("op", "}"); return ("block", body)

    def stmt(self):
        k, v = self.peek()
        if k == "kw" and v == "print":
            self.eat(); e = self.expr(); self.eat("op", ";"); return ("print", e)
        if k == "kw" and v == "while":
            self.eat(); self.eat("op", "("); c = self.expr(); self.eat("op", ")"); return ("while", c, self.block())
        if k == "kw" and v == "if":
            self.eat(); self.eat("op", "("); c = self.expr(); self.eat("op", ")"); th = self.block(); el = None
            if self.peek() == ("kw", "else"): self.eat(); el = self.block()
            return ("if", c, th, el)
        name = self.eat("id"); self.eat("op", "="); e = self.expr(); self.eat("op", ";"); return ("assign", name, e)

    def expr(self, lvl=0):
        if lvl == len(self.PREC): return self.unary()
        lhs = self.expr(lvl + 1)
        while self.peek()[0] == "op" and self.peek()[1] in self.PREC[lvl]:
            op = self.eat(); lhs = ("bin", op, lhs, self.expr(lvl + 1))
        return lhs

    def unary(self):
        k, v = self.peek()
        if (k, v) == ("op", "-"): self.eat(); return ("bin", "-", ("num", 0), self.unary())
        if (k, v) == ("op", "("): self.eat(); e = self.expr(); self.eat("op", ")"); return e
        if k == "num": self.eat(); return ("num", v)
        if k == "id": self.eat(); return ("var", v)
        raise SyntaxError(f"unexpected {k} {v}")


# ---------------------------------------------------------------- IR generation
# Instructions (tuples):
#   ("const", dst, int) ("mov", dst, src) ("bin", op, dst, a, b) ("print", a)
#   ("label", L) ("jmp", L) ("jz", a, L)
# operands are variable names (str) or ints (after constant propagation)
class IRGen:
    def __init__(self): self.code, self.nt, self.nl = [], 0, 0
    def temp(self): self.nt += 1; return f"t{self.nt}"
    def label(self): self.nl += 1; return f"L{self.nl}"

    def gen(self, node):
        kind = node[0]
        if kind == "block":
            for s in node[1]: self.gen(s)
        elif kind == "assign":
            v = self.expr(node[2]); self.code.append(("mov", node[1], v))
        elif kind == "print":
            self.code.append(("print", self.expr(node[1])))
        elif kind == "while":
            head, done = self.label(), self.label()
            self.code.append(("label", head)); c = self.expr(node[1]); self.code.append(("jz", c, done))
            self.gen(node[2]); self.code.append(("jmp", head)); self.code.append(("label", done))
        elif kind == "if":
            els, done = self.label(), self.label(); c = self.expr(node[1]); self.code.append(("jz", c, els))
            self.gen(node[2]); self.code.append(("jmp", done)); self.code.append(("label", els))
            if node[3]: self.gen(node[3])
            self.code.append(("label", done))

    def expr(self, e):
        if e[0] == "num":
            t = self.temp(); self.code.append(("const", t, e[1])); return t
        if e[0] == "var": return e[1]
        a = self.expr(e[2]); b = self.expr(e[3]); t = self.temp(); self.code.append(("bin", e[1], t, a, b)); return t


def compile_to_ir(src):
    g = IRGen(); g.gen(Parser(lex(src)).program()); return g.code


# ---------------------------------------------------------------- semantics shared by VM, folding and C
MASK = (1 << 64) - 1


def wrap(v):
    v &= MASK; return v - (1 << 64) if v >> 63 else v


def binop(op, a, b):
    if op == "+": return wrap(a + b)
    if op == "-": return wrap(a - b)
    if op == "*": return wrap(a * b)
    if op == "shl": return wrap(a << b)
    if op in ("/", "%"):
        if b == 0: raise ZeroDivisionError("division by zero")
        q = abs(a) // abs(b); q = q if (a >= 0) == (b >= 0) else -q
        return wrap(q) if op == "/" else wrap(a - b * q)
    return int({"<": a < b, ">": a > b, "<=": a <= b, ">=": a >= b, "==": a == b, "!=": a != b}[op])


CYCLES = {"const": 1, "mov": 1, "print": 1, "jmp": 1, "jz": 1, "load": 4, "store": 4,
          "+": 1, "-": 1, "shl": 1, "<": 1, ">": 1, "<=": 1, ">=": 1, "==": 1, "!=": 1, "*": 3, "/": 20, "%": 20}


def run_vm(code, max_steps=50_000_000):
    """Interpret IR. Returns (outputs, dynamic instruction count, weighted cycles, op histogram)."""
    labels = {ins[1]: i for i, ins in enumerate(code) if ins[0] == "label"}
    env, mem, out = {}, {}, []; pc = steps = cycles = 0; hist = {}
    val = lambda x: x if isinstance(x, int) else env.get(x, 0)
    n = len(code)
    while pc < n:
        ins = code[pc]; op = ins[0]
        if op == "label": pc += 1; continue
        steps += 1
        if steps > max_steps: raise RuntimeError("step limit")
        key = ins[1] if op == "bin" else op; cycles += CYCLES[key]; hist[key] = hist.get(key, 0) + 1
        if op == "const": env[ins[1]] = ins[2]
        elif op == "mov": env[ins[1]] = val(ins[2])
        elif op == "bin": env[ins[2]] = binop(ins[1], val(ins[3]), val(ins[4]))
        elif op == "print": out.append(val(ins[1]))
        elif op == "jmp": pc = labels[ins[1]]; continue
        elif op == "jz":
            if val(ins[1]) == 0: pc = labels[ins[2]]; continue
        elif op == "load": env[ins[1]] = mem.get(ins[2], 0)
        elif op == "store": mem[ins[1]] = val(ins[2])
        pc += 1
    return out, steps, cycles, hist


def static_size(code): return sum(1 for i in code if i[0] != "label")


# ---------------------------------------------------------------- C back end
def to_c(code):
    names = sorted({x for ins in code for x in ins[1:] if isinstance(x, str) and not x.startswith("L") and x not in
                    ("+", "-", "*", "/", "%", "<", ">", "<=", ">=", "==", "!=", "shl")} | {ins[1] for ins in code if ins[0] in ("const", "mov")}
                   | {ins[2] for ins in code if ins[0] == "bin"})
    names = [n for n in names if not (n.startswith("L") and n[1:].isdigit())]
    o = lambda x: f"{x}LL" if isinstance(x, int) else x
    lines = ["#include <stdio.h>", "#include <stdint.h>", "int main(void){"]
    if names: lines.append("  int64_t " + ", ".join(f"{n}=0" for n in names) + ";")
    for ins in code:
        k = ins[0]
        if k == "label": lines.append(f"{ins[1]}:;")
        elif k == "const": lines.append(f"  {ins[1]} = {o(ins[2])};")
        elif k == "mov": lines.append(f"  {ins[1]} = {o(ins[2])};")
        elif k == "bin":
            op = "<<" if ins[1] == "shl" else ins[1]
            if ins[1] in ("+", "-", "*", "shl"):   # wrap-around like the VM (unsigned arithmetic, no UB)
                lines.append(f"  {ins[2]} = (int64_t)((uint64_t){o(ins[3])} {op} (uint64_t){o(ins[4])});")
            else:
                lines.append(f"  {ins[2]} = {o(ins[3])} {op} {o(ins[4])};")
        elif k == "print": lines.append(f'  printf("%lld\\n", (long long){o(ins[1])});')
        elif k == "jmp": lines.append(f"  goto {ins[1]};")
        elif k == "jz": lines.append(f"  if ({o(ins[1])} == 0) goto {ins[2]};")
    lines.append("  return 0;\n}"); return "\n".join(lines)
