"""Tests for licm_solution.licm. Run: python3 /home/claude/tc_haiku/test_licm.py"""
import copy
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minic import compile_to_ir, run_vm  # noqa: E402
from licm_solution import licm  # noqa: E402


def check_wellformed(code):
    labels = [i[1] for i in code if i[0] == "label"]
    assert len(labels) == len(set(labels)), "duplicate labels"
    for i in code:
        if i[0] == "jmp":
            assert i[1] in labels, i
        if i[0] == "jz":
            assert i[2] in labels, i


def run_both(src):
    code = compile_to_ir(src)
    snapshot = copy.deepcopy(code)
    new = licm(code)
    assert code == snapshot, "input must not be mutated"
    check_wellformed(new)
    return run_vm(code), run_vm(new), new


def test_invariant_multiply_hoisted():
    src = """
    a = 3; b = 4; i = 0; s = 0;
    while (i < 5) { x = a * b + i; s = s + x / 2; print x; i = i + 1; }
    print s;
    """
    (out0, _, _, h0), (out1, _, _, h1), _ = run_both(src)
    assert out0 == out1
    assert h0["*"] == 5 and h1["*"] == 1, (h0["*"], h1["*"])


def test_division_by_variable_not_hoisted():
    src = """
    a = 100; d = 7; i = 0;
    while (i < 3) { q = a / d; r = a % d; print q + r; i = i + 1; }
    """
    (out0, _, _, h0), (out1, _, _, h1), _ = run_both(src)
    assert out0 == out1
    assert h1["/"] == 3 and h1["%"] == 3, "non-constant divisor must stay in the loop"


def test_division_by_constant_hoisted():
    src = """
    a = 100; i = 0;
    while (i < 3) { q = a / 7; print q; i = i + 1; }
    """
    (out0, _, _, _), (out1, _, _, h1), _ = run_both(src)
    assert out0 == out1
    assert h1["/"] == 1, "division by a nonzero constant may be hoisted"


def test_division_by_zero_never_hoisted():
    # The loop never runs; hoisting 1/0 would raise ZeroDivisionError in the preheader.
    src = """
    i = 0;
    while (i < 0) { q = 1 / 0; print q; i = i + 1; }
    print 42;
    """
    (out0, _, _, _), (out1, _, _, _), _ = run_both(src)
    assert out0 == out1 == [42]


def test_loop_variant_not_hoisted():
    src = """
    i = 0; s = 0;
    while (i < 4) { s = s + i * 2; i = i + 1; }
    print s;
    """
    (out0, _, _, _), (out1, _, _, _), _ = run_both(src)
    assert out0 == out1 == [12]


def test_nested_loops():
    src = """
    a = 2; b = 5; i = 0;
    while (i < 3) {
        j = 0;
        while (j < 4) { k = (a + 1) * (b + 2) + j; print k; j = j + 1; }
        i = i + 1;
    }
    """
    (out0, _, _, h0), (out1, _, _, h1), _ = run_both(src)
    assert out0 == out1
    assert h1["*"] < h0["*"], (h0["*"], h1["*"])


def test_if_inside_loop():
    src = """
    a = 6; i = 0; s = 0;
    while (i < 6) {
        if (i < 3) { s = s + a * 2; } else { s = s - a * 3; }
        i = i + 1;
    }
    print s;
    """
    (out0, _, _, _), (out1, _, _, _), _ = run_both(src)
    assert out0 == out1


def random_program(rng):
    """Random MiniC program with bounded nested loops, ifs, and divisions by safe divisors."""
    counter = [0]
    vars_ = ["a", "b", "c"]

    def expr(d=0):
        r = rng.random()
        if d > 2 or r < 0.3:
            return str(rng.randint(-5, 9)) if rng.random() < 0.5 else rng.choice(vars_)
        op = rng.choice(["+", "-", "*", "/", "%", "<", "=="])
        lhs, rhs = expr(d + 1), expr(d + 1)
        if op in ("/", "%"):
            rhs = rng.choice([str(rng.randint(1, 9)), f"(({rhs}) * ({rhs}) + 1)"])
        return f"(({lhs}) {op} ({rhs}))"

    def stmts(depth, n, loop_vars):
        out = []
        for _ in range(n):
            r = rng.random()
            if r < 0.35:
                out.append(f"{rng.choice(vars_)} = {expr()};")
            elif r < 0.5:
                out.append(f"print {expr()};")
            elif r < 0.65 and depth < 2:
                out.append(f"if ({expr()}) {{ {' '.join(stmts(depth + 1, rng.randint(1, 3), loop_vars))} }}"
                           f" else {{ {' '.join(stmts(depth + 1, rng.randint(1, 2), loop_vars))} }}")
            elif depth < 2:
                counter[0] += 1
                c = f"i{counter[0]}"
                bound = rng.randint(0, 4)
                body = stmts(depth + 1, rng.randint(1, 4), loop_vars + [c])
                out.append(f"{c} = 0; while ({c} < {bound}) {{ {' '.join(body)} {c} = {c} + 1; }}")
            else:
                out.append(f"print {rng.choice(vars_)};")
        return out

    init = f"a = {rng.randint(-3, 7)}; b = {rng.randint(-3, 7)}; c = {rng.randint(-3, 7)};"
    return init + " " + " ".join(stmts(0, rng.randint(2, 6), []))


def test_random_differential():
    rng = random.Random(12345)
    checked = changed = 0
    for _ in range(400):
        src = random_program(rng)
        code = compile_to_ir(src)
        new = licm(code)
        check_wellformed(new)
        out0 = run_vm(code)[0]
        out1 = run_vm(new)[0]
        assert out0 == out1, f"mismatch on program:\n{src}"
        checked += 1
        changed += new != code
    print(f"  random programs checked: {checked}, transformed: {changed}")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"all {len(tests)} tests passed")
