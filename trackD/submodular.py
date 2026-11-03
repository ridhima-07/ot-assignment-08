"""Track D1: submodular maximisation.

Objective: weighted coverage for placing k sensors (hospital-ward Wi-Fi/patient-monitor
access points) over demand points. f(S) = sum_j w_j * [j covered by some s in S].
Weighted coverage is monotone submodular, so greedy gets >= (1 - 1/e) OPT under a cardinality constraint.

We do not just assume this: we (1) test diminishing returns and monotonicity empirically,
(2) compare greedy with the exact optimum (exhaustive), (3) count oracle calls for lazy greedy,
(4) show a NON-submodular objective (pairwise complementarity) where the test fails and the
guarantee must not be claimed.
"""
import sys, os, itertools, heapq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from common.util import save_json, timer, mean_std

T = "trackD"


def instance(n_sites=20, n_points=200, radius=0.22, seed=0):
    rng = np.random.default_rng(seed)
    sites = rng.random((n_sites, 2)); pts = rng.random((n_points, 2)); w = rng.integers(1, 10, n_points)
    cover = np.linalg.norm(sites[:, None, :] - pts[None, :, :], axis=2) < radius
    return cover, w


class Coverage:
    def __init__(self, cover, w): self.cover, self.w, self.calls = cover, w, 0
    def __call__(self, S):
        self.calls += 1
        if not S: return 0.0
        return float(self.w[np.any(self.cover[list(S)], axis=0)].sum())


class Complementary:
    """f(S) = coverage(S) + bonus for specific PAIRS chosen together (supermodular part)."""
    def __init__(self, cover, w, pairs, bonus):
        self.cov = Coverage(cover, w); self.pairs, self.bonus, self.calls = pairs, bonus, 0
    def __call__(self, S):
        self.calls += 1; s = set(S)
        return self.cov(S) + self.bonus * sum(a in s and b in s for a, b in self.pairs)


def test_submodular(f, n, trials=3000, rng=None):
    """Check f(A+e)-f(A) >= f(B+e)-f(B) for random A subset of B, e not in B; and monotonicity."""
    rng = rng or np.random.default_rng(0); viol = mono = 0; worst = 0.0
    for _ in range(trials):
        B = set(rng.choice(n, rng.integers(1, n // 2), replace=False).tolist())
        A = set(x for x in B if rng.random() < 0.5)
        rest = [x for x in range(n) if x not in B]
        if not rest: continue
        e = rest[rng.integers(len(rest))]
        da, db = f(A | {e}) - f(A), f(B | {e}) - f(B)
        if da < db - 1e-9: viol += 1; worst = max(worst, db - da)
        if db < -1e-9: mono += 1
    return {"trials": trials, "diminishing_returns_violations": viol, "monotonicity_violations": mono, "worst_violation": worst}


def greedy(f, n, k):
    S = []
    for _ in range(k):
        base = f(S); best = max((x for x in range(n) if x not in S), key=lambda x: f(S + [x]) - base)
        S.append(best)
    return S


def lazy_greedy(f, n, k):
    """Minoux's accelerated greedy: stale marginal gains are upper bounds (valid only if submodular)."""
    S = []; base = 0.0
    heap = [(-f([x]), x) for x in range(n)]; heapq.heapify(heap)
    while len(S) < k:
        neg, x = heapq.heappop(heap); gain = f(S + [x]) - base
        if not heap or gain >= -heap[0][0] - 1e-12:
            S.append(x); base += gain
        else:
            heapq.heappush(heap, (-gain, x))
    return S


def exhaustive(f, n, k):
    return max(itertools.combinations(range(n), k), key=lambda S: f(list(S)))


def run():
    out = {}
    cover, w = instance()
    f = Coverage(cover, w); out["coverage_property_test"] = test_submodular(f, 20)
    rows = {}
    for k in [3, 4, 5]:
        ratios, calls_g, calls_l = [], [], []
        for seed in range(5):
            cv, ww = instance(seed=seed); f1, f2, f3 = Coverage(cv, ww), Coverage(cv, ww), Coverage(cv, ww)
            with timer() as tg: Sg = greedy(f1, 20, k)
            with timer() as tl: Sl = lazy_greedy(f2, 20, k)
            with timer() as te: So = exhaustive(f3, 20, k)
            assert abs(f1(Sg) - f2(Sl)) < 1e-9, "lazy greedy must match greedy"
            ratios.append(f1(Sg) / f1(list(So))); calls_g.append(f1.calls); calls_l.append(f2.calls)
        rows[k] = {"greedy/OPT_mean": mean_std(ratios)[0], "greedy/OPT_min": float(min(ratios)),
                   "bound_1-1/e": 1 - 1 / np.e, "bound_holds": bool(min(ratios) >= 1 - 1 / np.e),
                   "oracle_calls_greedy": float(np.mean(calls_g)), "oracle_calls_lazy": float(np.mean(calls_l))}
    out["coverage_greedy_vs_opt"] = rows
    # larger instance: exact via ILP-free exhaustive is impossible; report lazy-greedy savings only
    cv, ww = instance(n_sites=300, n_points=3000, radius=0.08, seed=1); fa, fb = Coverage(cv, ww), Coverage(cv, ww)
    with timer() as t1: a = greedy(fa, 300, 25)
    with timer() as t2: b = lazy_greedy(fb, 300, 25)
    out["large_n300_k25"] = {"greedy_calls": fa.calls, "lazy_calls": fb.calls, "greedy_secs": t1["s"], "lazy_secs": t2["s"],
                             "same_value": abs(fa(a) - fb(b)) < 1e-9}
    # non-submodular counter-example
    rng = np.random.default_rng(3); pairs = [tuple(rng.choice(20, 2, replace=False)) for _ in range(6)]
    g = Complementary(cover, w, pairs, bonus=120.0)
    out["complementary_property_test"] = test_submodular(g, 20)
    rat = []
    for k in [3, 4, 5]:
        Sg = greedy(g, 20, k); So = exhaustive(g, 20, k); rat.append(g(Sg) / g(list(So)))
    out["complementary_greedy/OPT_by_k"] = rat
    out["note"] = "Guarantee claimed only for the coverage objective, whose property test shows zero violations."
    return out


if __name__ == "__main__":
    r = run(); save_json(T, "submodular.json", r)
    import json; print(json.dumps(r, indent=1, default=str))
