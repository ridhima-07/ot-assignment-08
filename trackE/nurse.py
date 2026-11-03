"""Track E: hospital nurse rostering (main real-process case study).

PROCESS. A ward in-charge builds a 14-day roster for N nurses over three shifts
(M = 07-14, E = 14-21, N = 21-07). Done by hand it takes hours, routinely leaves shifts
under-staffed or without a senior nurse, ignores leave requests and loads night shifts on a
few people. We model the decision explicitly and compare a manual-style baseline with
three optimisation methods. All instances are synthetic; generator parameters are in GEN.

DECISION VARIABLES  x[n,d,s] in {0,1}: nurse n works shift s on day d.
HARD CONSTRAINTS
  H1 at most one shift per nurse per day
  H2 no M or E shift the day after an N shift (rest period)
  H3 contracted load: full-time <= 10 shifts / 14 days, part-time <= 6
  H4 at most 3 consecutive nights
  H5 approved leave days are not worked
SOFT CONSTRAINTS (objective, minimise)
  S1 understaffing: 100 per missing nurse below demand[d,s]
  S2 no senior nurse on a shift: 60 per shift
  S3 day-off requests worked: 10 each
  S4 night-shift fairness: 5 * (max nights - min nights) over full-time nurses
  S5 overstaffing: 2 per nurse above demand (cost of idle staff)
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
from common.util import save_json, outdir, plt, mean_std, md_table

T = "trackE"
D, S = 14, 3
SH = ["M", "E", "N"]
W = dict(under=100, senior=60, request=10, fair=5, over=2)
GEN = dict(senior_frac=0.3, pt_frac=0.25, demand_M=(4, 6), demand_E=(3, 5), demand_N=(3, 3), leave_prob=0.04, request_prob=0.2)  # ~97% of contracted capacity: a tight ward


def instance(n_nurses=20, seed=0):
    rng = np.random.default_rng(seed); scale = n_nurses / 20
    senior = rng.random(n_nurses) < GEN["senior_frac"]; senior[:2] = True
    pt = rng.random(n_nurses) < GEN["pt_frac"]; pt[senior] = False
    cap = np.where(pt, 6, 10)
    dem = np.zeros((D, S), int)
    for d in range(D):
        wk = d % 7 >= 5
        for s, key in enumerate(["demand_M", "demand_E", "demand_N"]):
            lo, hi = GEN[key]; dem[d, s] = max(1, int(round(scale * (lo if wk else hi))))
    leave = rng.random((n_nurses, D)) < GEN["leave_prob"]
    request = (rng.random((n_nurses, D)) < GEN["request_prob"]) & ~leave
    return dict(n=n_nurses, senior=senior, pt=pt, cap=cap, demand=dem, leave=leave, request=request)


# ---------------- independent checker (used for EVERY method) ----------------
def evaluate(I, x):
    n = I["n"]; x = x.astype(int); v = {}
    v["H1_multi_shift_days"] = int((x.sum(2) > 1).sum())
    v["H2_rest_violations"] = int(sum(x[i, d, 2] * (x[i, d + 1, 0] + x[i, d + 1, 1]) for i in range(n) for d in range(D - 1)))
    v["H3_over_contract"] = int(np.maximum(x.sum((1, 2)) - I["cap"], 0).sum())
    run4 = 0
    for i in range(n):
        for d in range(D - 3): run4 += int(x[i, d:d + 4, 2].sum() == 4)
    v["H4_4plus_nights"] = run4
    v["H5_leave_worked"] = int((x.sum(2) * I["leave"]).sum())
    staff = x.sum(0)
    under = np.maximum(I["demand"] - staff, 0).sum(); over = np.maximum(staff - I["demand"], 0).sum()
    no_senior = int(((x * I["senior"][:, None, None]).sum(0) == 0).sum())
    req = int((x.sum(2) * I["request"]).sum())
    nights = x[~I["pt"], :, 2].sum(1); fair = int(nights.max() - nights.min()) if len(nights) else 0
    cost = W["under"] * under + W["senior"] * no_senior + W["request"] * req + W["fair"] * fair + W["over"] * over
    hard = sum(val for k, val in v.items() if k.startswith("H"))
    return dict(cost=int(cost), hard_violations=hard, understaffed=int(under), no_senior_shifts=no_senior,
                requests_worked=req, night_spread=fair, overstaffed=int(over), **v)


def feasible_to_add(I, x, i, d, s):
    if I["leave"][i, d] or x[i, d].sum() > 0: return False
    if x[i].sum() >= I["cap"][i]: return False
    if s < 2 and d > 0 and x[i, d - 1, 2]: return False
    if s == 2 and d + 1 < D and (x[i, d + 1, 0] or x[i, d + 1, 1]): return False
    if s == 2:
        # length of the run of consecutive nights that would CONTAIN day d (look both ways).
        # (Bug fixed: the original check only looked at the 3 nights before d, which is enough for
        #  greedy's day-ordered filling but not for annealing's random moves.)
        lo = d
        while lo > 0 and x[i, lo - 1, 2]: lo -= 1
        hi = d
        while hi + 1 < D and x[i, hi + 1, 2]: hi += 1
        if hi - lo + 1 > 3: return False
    return True


# ---------------- baseline: how the roster is made by hand ----------------
def manual_baseline(I):
    """Day by day, shift by shift, fill the shift from the top of the staff list with whoever is
    still available (alphabetical / seniority list order). No look-ahead, no requests, no fairness."""
    x = np.zeros((I["n"], D, S), int)
    for d in range(D):
        for s in range(S):
            need = I["demand"][d, s]
            for i in range(I["n"]):
                if need == 0: break
                if feasible_to_add(I, x, i, d, s): x[i, d, s] = 1; need -= 1
    return x


# ---------------- method 1: greedy with scoring ----------------
def greedy(I):
    """Hardest shifts first (nights, then weekend days); for each slot pick the feasible nurse with
    the lowest score (requests, current night load, used fraction of contract), and secure a senior first."""
    x = np.zeros((I["n"], D, S), int)
    order = sorted([(d, s) for d in range(D) for s in range(S)], key=lambda t: (-(t[1] == 2), -(t[0] % 7 >= 5), t[0]))
    for d, s in order:
        for slot in range(I["demand"][d, s]):
            cands = [i for i in range(I["n"]) if feasible_to_add(I, x, i, d, s)]
            if not cands: break
            has_senior = (x[:, d, s] * I["senior"]).sum() > 0
            def score(i):
                return (0 if (has_senior or I["senior"][i]) else 1,          # get a senior on first
                        I["request"][i, d],                                   # avoid requested days off
                        x[i, :, 2].sum() if s == 2 else 0,                    # spread nights
                        x[i].sum() / I["cap"][i])                             # balance contract use
            x[min(cands, key=score), d, s] = 1
    return x


# ---------------- method 2: ILP (exact up to a time limit) ----------------
def ilp(I, time_limit=60):
    n = I["n"]; nx_ = n * D * S
    idx = lambda i, d, s: (i * D + d) * S + s
    # extra vars: under[d,s], over[d,s], nosenior[d,s], nmax, nmin
    u0 = nx_; o0 = u0 + D * S; z0 = o0 + D * S; nmax = z0 + D * S; nmin = nmax + 1; nv = nmin + 1
    c = np.zeros(nv)
    for d in range(D):
        for s in range(S):
            c[u0 + d * S + s] = W["under"]; c[o0 + d * S + s] = W["over"]; c[z0 + d * S + s] = W["senior"]
            for i in range(n): c[idx(i, d, s)] += W["request"] * I["request"][i, d]
    c[nmax] = W["fair"]; c[nmin] = -W["fair"]
    rows = []; lo = []; hi = []
    def add(coefs, l, h):
        rows.append(coefs); lo.append(l); hi.append(h)
    for i in range(n):
        for d in range(D):
            add({idx(i, d, s): 1 for s in range(S)}, 0, 0 if I["leave"][i, d] else 1)                 # H1 + H5
            if d + 1 < D:
                for s2 in [0, 1]: add({idx(i, d, 2): 1, idx(i, d + 1, s2): 1}, 0, 1)                    # H2
            if d + 3 < D: add({idx(i, d + k, 2): 1 for k in range(4)}, 0, 3)                           # H4
        add({idx(i, d, s): 1 for d in range(D) for s in range(S)}, 0, I["cap"][i])                     # H3
        if not I["pt"][i]:
            add({**{idx(i, d, 2): 1 for d in range(D)}, nmax: -1}, -np.inf, 0)                          # nights <= nmax
            add({**{idx(i, d, 2): 1 for d in range(D)}, nmin: -1}, 0, np.inf)                           # nights >= nmin
    for d in range(D):
        for s in range(S):
            k = d * S + s; dem = I["demand"][d, s]
            add({**{idx(i, d, s): 1 for i in range(n)}, u0 + k: 1, o0 + k: -1}, dem, dem)               # staff + under - over = demand
            add({**{idx(i, d, s): 1 for i in range(n) if I["senior"][i]}, z0 + k: 1}, 1, np.inf)        # senior or flag
    A = lil_matrix((len(rows), nv))
    for r, coefs in enumerate(rows):
        for j, v in coefs.items(): A[r, j] = v
    integ = np.r_[np.ones(nx_), np.zeros(2 * D * S), np.ones(D * S), 0, 0]
    ub = np.r_[np.ones(nx_), np.full(2 * D * S, np.inf), np.ones(D * S), D, D]
    t0 = time.perf_counter()
    r = milp(c, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=integ, bounds=Bounds(0, ub),
             options={"time_limit": time_limit, "mip_rel_gap": 0})
    secs = time.perf_counter() - t0
    x = np.round(r.x[:nx_]).astype(int).reshape(n, D, S)
    return x, {"status": r.message, "secs": secs, "objective": float(r.fun),
               "dual_bound": float(getattr(r, "mip_dual_bound", np.nan)), "gap": float(getattr(r, "mip_gap", np.nan)),
               "n_vars": nv, "n_constraints": len(rows)}


# ---------------- method 3: simulated annealing (local search) ----------------
def soft_cost(I, x):
    """Vectorised soft cost; valid only for rosters that satisfy every hard constraint
    (SA moves guarantee that, and the final roster is re-checked by evaluate())."""
    staff = x.sum(0); dem = I["demand"]
    nights = x[~I["pt"], :, 2].sum(1)
    return int(W["under"] * np.maximum(dem - staff, 0).sum() + W["over"] * np.maximum(staff - dem, 0).sum()
               + W["senior"] * ((x * I["senior"][:, None, None]).sum(0) == 0).sum()
               + W["request"] * (x.sum(2) * I["request"]).sum() + W["fair"] * (nights.max() - nights.min()))


def anneal(I, x0, iters=40000, T0=40.0, seed=0):
    """Moves keep every hard constraint satisfied: (a) reassign a shift to another nurse,
    (b) add a nurse to a shift, (c) remove a nurse from a shift. Cost = checker cost."""
    rng = np.random.default_rng(seed); x = x0.copy(); cur = soft_cost(I, x); best, bx = cur, x.copy()
    for k in range(iters):
        Tk = T0 * (1 - k / iters) + 0.05
        d, s = rng.integers(D), rng.integers(S); y = x.copy(); mv = rng.random()
        workers = np.flatnonzero(y[:, d, s]); others = [i for i in range(I["n"]) if not y[i, d, s]]
        if mv < 0.6 and len(workers):
            i = rng.choice(workers); y[i, d, s] = 0; j = rng.choice(others)
            if not feasible_to_add(I, y, j, d, s): continue
            y[j, d, s] = 1
        elif mv < 0.85:
            j = rng.choice(others)
            if not feasible_to_add(I, y, j, d, s): continue
            y[j, d, s] = 1
        elif len(workers):
            y[rng.choice(workers), d, s] = 0
        else:
            continue
        new = soft_cost(I, y)
        if new <= cur or rng.random() < np.exp(-(new - cur) / Tk):
            x, cur = y, new
            if cur < best: best, bx = cur, x.copy()
    return bx


def run_all(sizes=(10, 20, 40, 80), seeds=5, ilp_limit=60):
    out = {}
    for n in sizes:
        rows = {m: [] for m in ["Manual baseline", "Greedy", "Simulated annealing", "ILP (HiGHS)"]}; ilp_info = []
        for sd in range(seeds):
            I = instance(n, sd)
            t = time.perf_counter(); xb = manual_baseline(I); tb = time.perf_counter() - t
            t = time.perf_counter(); xg = greedy(I); tg = time.perf_counter() - t
            t = time.perf_counter(); xa = anneal(I, xg, seed=sd); ta = time.perf_counter() - t
            xi, info = ilp(I, ilp_limit); ilp_info.append(info)
            for m, x, tt in [("Manual baseline", xb, tb), ("Greedy", xg, tg), ("Simulated annealing", xa, ta), ("ILP (HiGHS)", xi, info["secs"])]:
                e = evaluate(I, x); e["secs"] = tt; rows[m].append(e)
            # ILP objective must equal the independent checker's cost
            assert abs(evaluate(I, xi)["cost"] - info["objective"]) < 1e-6, f"ILP objective {info['objective']} != checker {evaluate(I, xi)['cost']}"
        out[n] = {m: {k: mean_std([e[k] for e in es])[0] for k in es[0]} for m, es in rows.items()}
        out[n]["ILP (HiGHS)"]["proved_optimal_runs"] = sum("Optimal" in i["status"] for i in ilp_info)
        out[n]["ILP (HiGHS)"]["model_size"] = f"{ilp_info[0]['n_vars']} vars, {ilp_info[0]['n_constraints']} constraints"
        out[n]["ILP (HiGHS)"]["mean_gap"] = float(np.nanmean([i["gap"] for i in ilp_info]))
        print(n, {m: (round(v["cost"]), round(v["secs"], 2)) for m, v in out[n].items()}, flush=True)
    return out


def sensitivity():
    """What if demand rises 25% (flu season)? Same 20 nurses."""
    res = {}
    for bump in [1.0, 1.25]:
        I = instance(20, 0); I["demand"] = np.ceil(I["demand"] * bump).astype(int)
        res[str(bump)] = {"Manual": evaluate(I, manual_baseline(I)), "ILP": evaluate(I, ilp(I, 60)[0])}
    return res


def plot(out):
    p = plt(); fig, axes = p.subplots(1, 2, figsize=(11, 4)); ms = ["Manual baseline", "Greedy", "Simulated annealing", "ILP (HiGHS)"]
    sizes = list(out)
    for m in ms:
        axes[0].plot(sizes, [out[s][m]["cost"] for s in sizes], "o-", label=m)
        axes[1].semilogy(sizes, [max(out[s][m]["secs"], 1e-4) for s in sizes], "o-", label=m)
    axes[0].set_xlabel("nurses on ward"); axes[0].set_ylabel("roster penalty (lower is better)"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("nurses on ward"); axes[1].set_ylabel("runtime (s)"); axes[1].legend(fontsize=8)
    fig.suptitle("Nurse rostering: quality vs runtime (mean of 5 instances)"); fig.tight_layout()
    fig.savefig(os.path.join(outdir(T), "E_quality_runtime.png")); p.close(fig)


def roster_figure():
    I = instance(20, 0); p = plt(); fig, axes = p.subplots(1, 2, figsize=(12, 5))
    for ax, (name, x) in zip(axes, [("Manual baseline", manual_baseline(I)), ("ILP", ilp(I, 60)[0])]):
        grid = np.zeros((I["n"], D))
        for s in range(S): grid += (s + 1) * x[:, :, s]
        import matplotlib; ax.imshow(grid, cmap=matplotlib.colormaps["viridis"].resampled(4), vmin=0, vmax=3, aspect="auto")
        ax.set_title(f"{name}: cost {evaluate(I, x)['cost']}"); ax.set_xlabel("day"); ax.set_ylabel("nurse (seniors first rows marked *)")
        ax.set_yticks(range(I["n"])); ax.set_yticklabels([("*" if I["senior"][i] else "") + str(i) for i in range(I["n"])], fontsize=6)
    fig.text(0.5, 0.01, "colour: dark = off, then Morning, Evening, Night (yellow)", ha="center")
    fig.tight_layout(); fig.savefig(os.path.join(outdir(T), "E_rosters.png")); p.close(fig)


if __name__ == "__main__":
    out = run_all()
    save_json(T, "nurse.json", {"results_by_ward_size": out, "sensitivity_demand": sensitivity(), "weights": W, "generator": GEN})
    plot(out); roster_figure()
    print(json.dumps(out, indent=1, default=str)[:4000])
