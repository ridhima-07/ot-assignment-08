"""Fast correctness tests for the claims the report relies on. Run: python3 -m pytest -q tests"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


# ---------- Track A ----------
def test_optimizers_converge_on_quadratic():
    from trackA.optimizers import GD, Momentum, NAG, Adam, RMSProp, AdaGrad, Newton, run
    A = np.diag([1.0, 10.0]); f = lambda x: 0.5 * x @ A @ x; g = lambda x: A @ x
    for opt in [GD(0.09), Momentum(0.05), NAG(0.05), Adam(0.1), RMSProp(0.01), AdaGrad(0.5)]:
        _, ls = run(opt, [1.0, 1.0], g, f, 3000)
        # RMSProp with a constant step behaves like sign descent near the optimum and oscillates
        # with amplitude ~lr, so f plateaus around lr^2 (~1e-4 here) instead of going to 0.
        assert ls[-1] < (1e-3 if isinstance(opt, RMSProp) else 1e-8), type(opt).__name__
    _, ls = run(Newton(1.0), [1.0, 1.0], g, f, 1, hess_fn=lambda x: A)
    assert ls[-1] < 1e-20


def test_backprop_matches_finite_differences():
    from trackA.ml_optim import MLP
    rng = np.random.default_rng(0); X = rng.normal(size=(30, 2)); y = (X[:, 0] > 0).astype(float)
    net = MLP(2, 5, rng); th = net.theta; g = net.backprop(th, X, y); e = 1e-6
    fd = np.array([(net.loss(th + e * d, X, y) - net.loss(th - e * d, X, y)) / (2 * e) for d in np.eye(len(th))])
    assert np.linalg.norm(g - fd) / np.linalg.norm(g + fd) < 1e-7


# ---------- Track B ----------
def test_knapsack_exact_methods_agree():
    from trackB.discrete import knap_instance, knap_dp, knap_ilp, knap_bnb, knap_greedy
    for s in range(5):
        v, w, C = knap_instance(25, np.random.default_rng(s))
        opt = knap_dp(v, w, C)
        assert knap_ilp(v, w, C) == opt and knap_bnb(v, w, C)[0] == opt and knap_greedy(v, w, C) <= opt


def test_qubo_ising_equal_cut():
    from trackB.discrete import maxcut_instance, maxcut_qubo, qubo_to_ising, cut_value
    G = maxcut_instance(6, seed=3); Q = maxcut_qubo(G); J, h, c = qubo_to_ising(Q)
    for bits in itertools.product([0, 1], repeat=6):
        x = np.array(bits); s = 1 - 2 * x
        assert abs(x @ Q @ x + cut_value(G, x)) < 1e-9 and abs(x @ Q @ x - (s @ J @ s + h @ s + c)) < 1e-9


# ---------- Track D ----------
def test_coverage_is_submodular_and_lazy_matches_greedy():
    from trackD.submodular import instance, Coverage, test_submodular, greedy, lazy_greedy
    cv, w = instance(n_sites=12, n_points=80); f = Coverage(cv, w)
    assert test_submodular(f, 12, trials=500)["diminishing_returns_violations"] == 0
    assert abs(f(greedy(f, 12, 4)) - f(lazy_greedy(f, 12, 4))) < 1e-9


# ---------- Track E ----------
def test_nurse_checker_and_ilp_feasible():
    from trackE.nurse import instance, evaluate, ilp, manual_baseline
    I = instance(10, 0); x, info = ilp(I, 30)
    e = evaluate(I, x); assert e["hard_violations"] == 0 and abs(e["cost"] - info["objective"]) < 1e-6
    assert e["cost"] <= evaluate(I, manual_baseline(I))["cost"]
    bad = x.copy(); bad[0, 0, :] = 1                         # three shifts on one day must be flagged
    assert evaluate(I, bad)["H1_multi_shift_days"] >= 1


# ---------- Track F ----------
def test_compiler_preserves_output_on_random_programs():
    from trackF.fuzz import fuzz
    assert fuzz(60)["mismatches"] == 0


def test_constant_folding_and_licm():
    from trackF.minic import compile_to_ir, run_vm
    from trackF.passes import optimise
    ir = compile_to_ir("n = 10; i = 0; s = 0; while (i < n) { k = 3 * 4; s = s + k; i = i + 1; } print s;")
    opt = optimise(ir)
    assert run_vm(opt)[0] == [120] and run_vm(opt)[1] < run_vm(ir)[1]
    assert not any(ins[0] == "bin" and ins[1] == "*" for ins in opt)   # 3*4 folded away
