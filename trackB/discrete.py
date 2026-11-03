"""Track B: discrete optimisation.

  B1 0/1 knapsack: greedy (ratio), dynamic programming, ILP (HiGHS), own branch-and-bound
  B2 identical-machine makespan scheduling P||Cmax: LPT greedy vs ILP vs branch-and-bound,
     with a check of Graham's 4/3 - 1/(3m) LPT guarantee
  B3 Max-Cut as QUBO and as Ising (binary-to-spin), verified on all 2^n assignments,
     solved by brute force, greedy local search, simulated annealing and ILP.
     The instance is saved for Track C (QAOA).
  B4 Constrained QUBO: choose exactly k features (relevance - redundancy) with a
     quadratic penalty P(sum x - k)^2; sweep P to show when the penalty is large enough.
"""
import sys, os, itertools, heapq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import networkx as nx
from scipy.optimize import milp, LinearConstraint, Bounds
from common.util import save_json, outdir, plt, timer, mean_std

T = "trackB"


# ---------------- B1 knapsack ----------------
def knap_instance(n, rng):
    w = rng.integers(10, 100, n); v = w + rng.integers(-5, 30, n)   # correlated -> harder
    return v.astype(int), w.astype(int), int(w.sum() // 2)


def knap_greedy(v, w, C):
    order = np.argsort(-v / w); tot = val = 0; take = []
    for i in order:
        if tot + w[i] <= C: tot += w[i]; val += v[i]; take.append(i)
    best1 = max((v[i] for i in range(len(v)) if w[i] <= C), default=0)   # max(greedy, best single) = 1/2-approx
    return max(val, best1)


def knap_dp(v, w, C):
    dp = np.zeros(C + 1, dtype=np.int64)
    for i in range(len(v)):
        dp[w[i]:] = np.maximum(dp[w[i]:], dp[:C + 1 - w[i]] + v[i])
    return int(dp[C])


def knap_ilp(v, w, C):
    n = len(v)
    r = milp(c=-v, constraints=LinearConstraint(w[None, :], 0, C), integrality=np.ones(n), bounds=Bounds(0, 1))
    return int(round(-r.fun))


def knap_bnb(v, w, C):
    """Depth-first B&B; bound = fractional (LP) relaxation on remaining items sorted by ratio."""
    order = np.argsort(-v / w); v, w = v[order], w[order]; n = len(v)
    best = [knap_greedy(v, w, C)]; nodes = [0]

    def bound(i, cap, val):
        while i < n and w[i] <= cap: cap -= w[i]; val += v[i]; i += 1
        return val + (v[i] * cap / w[i] if i < n else 0)

    def dfs(i, cap, val):
        nodes[0] += 1
        if val > best[0]: best[0] = val
        if i == n or bound(i, cap, val) <= best[0]: return
        if w[i] <= cap: dfs(i + 1, cap - w[i], val + v[i])
        dfs(i + 1, cap, val)
    sys.setrecursionlimit(10000); dfs(0, C, 0)
    return int(best[0]), nodes[0]


def b1_knapsack(sizes=(20, 50, 100, 200), seeds=5):
    out = {}
    for n in sizes:
        rows = {m: {"val": [], "t": []} for m in ["greedy", "dp", "ilp", "bnb"]}; nodes = []
        for s in range(seeds):
            v, w, C = knap_instance(n, np.random.default_rng(s))
            for m, fn in [("greedy", knap_greedy), ("dp", knap_dp), ("ilp", knap_ilp)]:
                with timer() as t: val = fn(v, w, C)
                rows[m]["val"].append(val); rows[m]["t"].append(t["s"])
            with timer() as t: val, nd = knap_bnb(v, w, C)
            rows["bnb"]["val"].append(val); rows["bnb"]["t"].append(t["s"]); nodes.append(nd)
        opt = np.array(rows["dp"]["val"])
        assert np.all(np.array(rows["ilp"]["val"]) == opt) and np.all(np.array(rows["bnb"]["val"]) == opt), "exact methods disagree"
        out[n] = {m: {"gap_%_mean": float(np.mean(100 * (opt - np.array(d["val"])) / opt)),
                      "ms_mean": 1000 * mean_std(d["t"])[0]} for m, d in rows.items()}
        out[n]["bnb_nodes_mean"] = float(np.mean(nodes)); out[n]["capacity_mean"] = None
    return out


# ---------------- B2 makespan ----------------
def lpt(p, m):
    loads = [(0, k) for k in range(m)]; heapq.heapify(loads)
    for j in np.argsort(-p):
        l, k = heapq.heappop(loads); heapq.heappush(loads, (l + p[j], k))
    return max(l for l, _ in loads)


def makespan_ilp(p, m):
    n = len(p); nv = n * m + 1          # x_{jk}, Cmax
    c = np.zeros(nv); c[-1] = 1
    A, lo, hi = [], [], []
    for j in range(n):                  # each job on exactly one machine
        row = np.zeros(nv); row[j * m:(j + 1) * m] = 1; A.append(row); lo.append(1); hi.append(1)
    for k in range(m):                  # load_k <= Cmax
        row = np.zeros(nv); row[k:n * m:m] = p; row[-1] = -1; A.append(row); lo.append(-np.inf); hi.append(0)
    integ = np.r_[np.ones(n * m), 0]
    r = milp(c, constraints=LinearConstraint(np.array(A), lo, hi), integrality=integ,
             bounds=Bounds(np.zeros(nv), np.r_[np.ones(n * m), np.inf]), options={"time_limit": 30})
    return float(r.fun), r.status == 0


def makespan_bnb(p, m):
    p = np.sort(p)[::-1]; best = [lpt(p, m)]; lb = max(p.max(), p.sum() / m); nodes = [0]
    loads = np.zeros(m)
    def dfs(j):
        nodes[0] += 1
        if best[0] <= lb + 1e-9: return
        if j == len(p):
            best[0] = min(best[0], loads.max()); return
        seen = set()
        for k in range(m):
            if loads[k] in seen: continue      # symmetry breaking: equal loads are interchangeable
            seen.add(loads[k])
            if loads[k] + p[j] >= best[0]: continue
            loads[k] += p[j]; dfs(j + 1); loads[k] -= p[j]
    dfs(0)
    return float(best[0]), nodes[0]


def b2_makespan(cases=((10, 3), (14, 4), (20, 5)), seeds=5):
    out = {}
    for n, m in cases:
        g, ti, tl, tb, ratio_ok, nodes = [], [], [], [], [], []
        for s in range(seeds):
            p = np.random.default_rng(s).integers(5, 50, n).astype(float)
            with timer() as t1: L = lpt(p, m)
            with timer() as t2: I, ok = makespan_ilp(p, m)
            with timer() as t3: B, nd = makespan_bnb(p, m)
            assert abs(I - B) < 1e-6 or not ok, f"ILP {I} vs B&B {B}"
            g.append(100 * (L - B) / B); tl.append(t1["s"]); ti.append(t2["s"]); tb.append(t3["s"]); nodes.append(nd)
            ratio_ok.append(L / B <= 4 / 3 - 1 / (3 * m) + 1e-12)
        out[f"n={n},m={m}"] = {"LPT_gap_%_mean": float(np.mean(g)), "LPT_gap_%_max": float(np.max(g)),
                               "graham_bound_holds_all": bool(all(ratio_ok)), "ms_LPT": 1000 * np.mean(tl),
                               "ms_ILP": 1000 * np.mean(ti), "ms_BnB": 1000 * np.mean(tb), "bnb_nodes_mean": float(np.mean(nodes))}
    return out


# ---------------- B3 Max-Cut QUBO / Ising ----------------
def maxcut_instance(n=8, p=0.5, seed=7):
    G = nx.gnp_random_graph(n, p, seed=seed)
    rng = np.random.default_rng(seed)
    for u, v in G.edges: G[u][v]["weight"] = 1.0       # unweighted for QAOA clarity
    return G


def maxcut_qubo(G):
    """minimise x^T Q x  (x in {0,1}^n) with  -cut(x) = sum_ij w_ij(2 x_i x_j - x_i - x_j)."""
    n = G.number_of_nodes(); Q = np.zeros((n, n))
    for u, v, d in G.edges(data=True):
        w = d["weight"]; Q[u, u] -= w; Q[v, v] -= w; Q[u, v] += w; Q[v, u] += w
    return Q


def qubo_to_ising(Q):
    """x = (1 - s)/2  ->  x^T Q x = s^T J s + h^T s + c   with J having zero diagonal."""
    n = len(Q); J = Q / 4; np.fill_diagonal(J, 0)
    h = -(Q.sum(1) + Q.sum(0)) / 4
    c = Q.sum() / 4 + np.trace(Q) / 4
    return J, h, c


def cut_value(G, x): return sum(d["weight"] for u, v, d in G.edges(data=True) if x[u] != x[v])


def greedy_local_search(G, rng):
    n = G.number_of_nodes(); x = rng.integers(0, 2, n)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            y = x.copy(); y[i] ^= 1
            if cut_value(G, y) > cut_value(G, x): x, improved = y, True
    return x


def simulated_annealing(Q, rng, iters=4000, T0=2.0):
    n = len(Q); x = rng.integers(0, 2, n); e = x @ Q @ x; best = (e, x.copy())
    for k in range(iters):
        Tk = T0 * (1 - k / iters) + 1e-3; i = rng.integers(n); y = x.copy(); y[i] ^= 1; ey = y @ Q @ y
        if ey < e or rng.random() < np.exp(-(ey - e) / Tk): x, e = y, ey
        if e < best[0]: best = (e, x.copy())
    return best[1]


def maxcut_ilp(G):
    """Standard linearisation: y_e <= x_u + x_v, y_e <= 2 - x_u - x_v, maximise sum w_e y_e."""
    n = G.number_of_nodes(); E = list(G.edges(data=True)); m = len(E)
    c = np.r_[np.zeros(n), -np.array([d["weight"] for *_, d in E])]
    A = []
    for k, (u, v, _) in enumerate(E):
        r1 = np.zeros(n + m); r1[n + k] = 1; r1[u] = -1; r1[v] = -1; A.append(r1)       # y - xu - xv <= 0
        r2 = np.zeros(n + m); r2[n + k] = 1; r2[u] = 1; r2[v] = 1; A.append(r2)         # y + xu + xv <= 2
    ub = np.array([0, 2] * m, dtype=float)
    r = milp(c, constraints=LinearConstraint(np.array(A), -np.inf, ub), integrality=np.ones(n + m), bounds=Bounds(0, 1))
    return -r.fun, np.round(r.x[:n]).astype(int)


def b3_maxcut():
    G = maxcut_instance(); n = G.number_of_nodes(); Q = maxcut_qubo(G); J, h, c = qubo_to_ising(Q)
    # verify QUBO and Ising energies against the direct cut for all 2^n assignments
    max_err = 0.0; best = (-1, None); all_cuts = []
    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits); s = 1 - 2 * x
        cut = cut_value(G, x); eq = x @ Q @ x; ei = s @ J @ s + h @ s + c
        max_err = max(max_err, abs(-cut - eq), abs(eq - ei)); all_cuts.append(cut)
        if cut > best[0]: best = (cut, x)
    rng = np.random.default_rng(0)
    gls = [cut_value(G, greedy_local_search(G, np.random.default_rng(s))) for s in range(20)]
    sa = [cut_value(G, simulated_annealing(Q, np.random.default_rng(s))) for s in range(20)]
    ilp_val, ilp_x = maxcut_ilp(G)
    save_json(T, "maxcut_instance.json", {"n": n, "edges": [[u, v] for u, v in G.edges], "optimum": best[0],
                                         "optimal_assignments": [list(map(int, b)) for b in itertools.product([0, 1], repeat=n)
                                                                 if cut_value(G, np.array(b)) == best[0]],
                                         "Q": Q.tolist(), "J": J.tolist(), "h": h.tolist(), "c": c})
    return {"n": n, "edges": G.number_of_edges(), "max_energy_mismatch_over_2^n": max_err,
            "brute_force_opt": best[0], "random_assignment_mean_cut": float(np.mean(all_cuts)),
            "greedy_local_search": {"mean": float(np.mean(gls)), "hit_opt_frac": float(np.mean(np.array(gls) == best[0]))},
            "simulated_annealing": {"mean": float(np.mean(sa)), "hit_opt_frac": float(np.mean(np.array(sa) == best[0]))},
            "ilp": ilp_val}


# ---------------- B4 penalty QUBO: choose exactly k features ----------------
def b4_penalty(k=6):   # k chosen so the unconstrained optimum (size 3) violates the constraint
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True); X = X[:, :10]
    rel = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(10)])
    red = np.abs(np.corrcoef(X.T)); np.fill_diagonal(red, 0)
    n = 10; base = -np.diag(rel) + 0.5 * red / 2      # x^T Q x = -sum rel_i x_i + 0.5 sum_{i<j} red_ij x_i x_j
    def build(P):
        # P (sum x - k)^2 = P[ sum_i x_i (1 - 2k) + 2 sum_{i<j} x_i x_j + k^2 ]
        Q = base.copy() + P * (np.ones((n, n)) - np.eye(n)) + np.diag(np.full(n, P * (1 - 2 * k)))
        return Q, P * k * k
    def true_obj(x): return -rel @ x + 0.5 * x @ (red / 2) @ x
    out = {}
    feasible_best = min((true_obj(np.array(b)), b) for b in itertools.product([0, 1], repeat=n) if sum(b) == k)
    for P in [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        Q, const = build(P)
        e, b = min((np.array(b) @ Q @ np.array(b) + const, b) for b in itertools.product([0, 1], repeat=n))
        out[str(P)] = {"chosen_size": int(sum(b)), "feasible": sum(b) == k,
                       "matches_constrained_opt": tuple(b) == tuple(feasible_best[1]), "subset": [i for i in range(n) if b[i]]}
    out["constrained_optimum_subset"] = [i for i in range(n) if feasible_best[1][i]]
    out["max_single_term_gain"] = float(rel.max())
    return out


if __name__ == "__main__":
    r = {"B1_knapsack": b1_knapsack(), "B2_makespan": b2_makespan(), "B3_maxcut": b3_maxcut(), "B4_penalty_qubo": b4_penalty()}
    save_json(T, "discrete.json", r)
    import json; print(json.dumps(r, indent=1, default=str))
