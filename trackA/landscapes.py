"""Track A, part 1: optimizers on synthetic landscapes.

Experiments
  A1 convergence on strongly convex / convex / non-convex objectives vs theory
  A2 saddle point: first-order methods escape, plain Newton is attracted to it
  A3 local vs global minima: basin statistics from a grid of starts
  A4 step-size sensitivity sweep
  A5 noisy gradients: constant vs decaying step, repeated seeds
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
np.seterr(all="ignore")  # divergence is detected explicitly in run(); silence overflow noise
from common.util import save_json, outdir, plt, seed_all, timer, mean_std, md_table
from trackA.optimizers import GD, Momentum, NAG, AdaGrad, RMSProp, Adam, Newton, run

T = "trackA"

# ---------- objectives ----------
theta = np.pi / 6
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
MU, L = 1.0, 50.0
A = R @ np.diag([MU, L]) @ R.T                       # strongly convex, kappa = 50
quad = dict(f=lambda x: 0.5 * x @ A @ x, g=lambda x: A @ x, H=lambda x: A, xstar=np.zeros(2))

quart = dict(f=lambda x: np.sum(x ** 4), g=lambda x: 4 * x ** 3,     # convex, not strongly convex at x*
             H=lambda x: np.diag(12 * x ** 2 + 1e-12), xstar=np.zeros(2))

def rosen_f(x): return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
def rosen_g(x): return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)])
def rosen_H(x): return np.array([[2 - 400 * (x[1] - 3 * x[0] ** 2), -400 * x[0]], [-400 * x[0], 200]])
rosen = dict(f=rosen_f, g=rosen_g, H=rosen_H, xstar=np.array([1.0, 1.0]))

# saddle at (0,0); minima at (0, +-sqrt(2)) with f = -1
def sad_f(x): return x[0] ** 2 - x[1] ** 2 + x[1] ** 4 / 4
def sad_g(x): return np.array([2 * x[0], -2 * x[1] + x[1] ** 3])
def sad_H(x): return np.array([[2.0, 0.0], [0.0, -2 + 3 * x[1] ** 2]])

# double well with tilt: local min near x=+0.96, global near x=-1.03
def dw_f(x): return (x[0] ** 2 - 1) ** 2 + 0.3 * x[0] + x[1] ** 2
def dw_g(x): return np.array([4 * x[0] * (x[0] ** 2 - 1) + 0.3, 2 * x[1]])


def make(name, lr):
    return {"GD": lambda: GD(lr), "Momentum": lambda: Momentum(lr, 0.9), "NAG": lambda: NAG(lr, 0.9),
            "AdaGrad": lambda: AdaGrad(lr), "RMSProp": lambda: RMSProp(lr), "Adam": lambda: Adam(lr)}[name]()


FIRST = ["GD", "Momentum", "NAG", "AdaGrad", "RMSProp", "Adam"]


def tuned_lr(obj, name, x0, iters, grid):
    """Pick the learning rate with lowest final loss (grid search, same budget for all)."""
    best = (np.inf, None)
    for lr in grid:
        _, ls = run(make(name, lr), x0, obj["g"], obj["f"], iters)
        fin = ls[-1] if np.isfinite(ls[-1]) else np.inf
        if fin < best[0]: best = (fin, lr)
    return best[1]


def a1_convergence():
    p = plt(); res = {}
    fig, axes = p.subplots(1, 3, figsize=(15, 4.2))
    grid = [10 ** e for e in np.linspace(-4, 0, 17)]
    for ax, (oname, obj, x0, iters) in zip(axes, [
            ("strongly convex quadratic (kappa=50)", quad, np.array([1.0, 1.0]), 300),
            ("convex quartic sum x^4", quart, np.array([1.0, -0.8]), 2000),
            ("non-convex Rosenbrock", rosen, np.array([-1.2, 1.0]), 5000)]):
        res[oname] = {}
        fstar = obj["f"](obj["xstar"])
        for n in FIRST:
            lr = tuned_lr(obj, n, x0, iters, grid)
            with timer() as t:
                traj, ls = run(make(n, lr), x0, obj["g"], obj["f"], iters)
            gap = np.maximum(ls - fstar, 1e-16)
            res[oname][n] = {"lr": lr, "final_gap": float(gap[-1]), "secs": t["s"],
                             "iters_to_1e-6": int(np.argmax(gap < 1e-6)) if np.any(gap < 1e-6) else None}
            ax.semilogy(gap, label=f"{n} (lr={lr:.2g})")
        nt = Newton(1.0, damping=1e-8)
        traj, ls = run(nt, x0, obj["g"], obj["f"], min(iters, 100), hess_fn=obj["H"])
        gap = np.maximum(ls - fstar, 1e-16)
        res[oname]["Newton"] = {"lr": 1.0, "final_gap": float(gap[-1]),
                                "iters_to_1e-6": int(np.argmax(gap < 1e-6)) if np.any(gap < 1e-6) else None}
        ax.semilogy(gap, "k--", lw=2, label="Newton")
        if obj is quad:   # theory: GD with lr=1/L contracts f-f* by (1-mu/L)^2 per step
            k = np.arange(iters + 1)
            ax.semilogy(quad["f"](x0) * (1 - MU / L) ** (2 * k), "r:", lw=2, label="GD bound (1-mu/L)^2k")
            _, lsg = run(GD(1 / L), x0, quad["g"], quad["f"], iters)
            res[oname]["GD@1/L_obeys_bound"] = bool(np.all(lsg <= quad["f"](x0) * (1 - MU / L) ** (2 * k) + 1e-12))
        if obj is quart:  # sublinear: fit slope on log-log for GD
            _, lsg = run(GD(res[oname]["GD"]["lr"]), x0, obj["g"], obj["f"], iters)
            k = np.arange(100, iters + 1)
            slope = np.polyfit(np.log(k), np.log(lsg[100:]), 1)[0]
            res[oname]["GD_loglog_slope"] = float(slope)
        ax.set_title(oname); ax.set_xlabel("iteration"); ax.set_ylabel("f(x) - f*")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(outdir(T), "A1_convergence.png")); p.close(fig)
    return res


def a2_saddle():
    """Start a hair away from the saddle along the unstable direction."""
    x0 = np.array([0.5, 1e-4]); iters = 200; out = {}
    p = plt(); fig, ax = p.subplots(figsize=(6, 5))
    xs = np.linspace(-1, 1, 200); ys = np.linspace(-2, 2, 200); X, Y = np.meshgrid(xs, ys)
    ax.contour(X, Y, X ** 2 - Y ** 2 + Y ** 4 / 4, levels=30, cmap="viridis", alpha=0.6)
    for n, opt in [("GD", GD(0.05)), ("Momentum", Momentum(0.05)), ("Adam", Adam(0.05)),
                   ("Newton", Newton(1.0)), ("Saddle-free Newton |H|", None)]:
        if opt is None:   # saddle-free Newton: use |H| (absolute eigenvalues)
            x = x0.copy(); traj = [x.copy()]
            for _ in range(iters):
                w, V = np.linalg.eigh(sad_H(x)); Habs = V @ np.diag(np.maximum(np.abs(w), 1e-3)) @ V.T
                x = x - np.linalg.solve(Habs, sad_g(x)); traj.append(x.copy())
            traj = np.array(traj)
        else:
            traj, _ = run(opt, x0, sad_g, sad_f, iters, hess_fn=sad_H if n == "Newton" else None)
        xf = traj[-1]; eig = np.linalg.eigvalsh(sad_H(xf))
        kind = "saddle" if eig.min() < -1e-6 else ("minimum" if eig.min() > 1e-6 else "degenerate")
        out[n] = {"final_x": xf.round(4).tolist(), "f": float(sad_f(xf)), "hessian_eigs": eig.round(3).tolist(),
                  "converged_to": kind}
        ax.plot(traj[:, 0], traj[:, 1], ".-", ms=3, label=f"{n} -> {kind}")
    ax.plot(0, 0, "rx", ms=12, mew=3); ax.legend(fontsize=7); ax.set_title("f = x^2 - y^2 + y^4/4 (saddle at origin)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir(T), "A2_saddle.png")); p.close(fig)
    return out


def a3_local_minima():
    starts = [np.array([x, y]) for x in np.linspace(-2, 2, 21) for y in np.linspace(-1, 1, 5)]
    xg = -1.0371  # global min x (root of 4x^3-4x+0.3 near -1)
    out = {}
    for n, lr in [("GD", 0.05), ("Momentum", 0.02), ("NAG", 0.02), ("Adam", 0.05), ("RMSProp", 0.02), ("AdaGrad", 0.3)]:
        hits = 0
        for s in starts:
            traj, _ = run(make(n, lr), s, dw_g, dw_f, 2000)
            hits += abs(traj[-1][0] - xg) < 0.05
        out[n] = {"lr": lr, "frac_global": hits / len(starts)}
    return out


def a4_stepsize():
    grid = [10 ** e for e in np.linspace(-4, 0.5, 19)]
    out = {}
    for oname, obj, x0, iters in [("quadratic", quad, np.array([1.0, 1.0]), 300),
                                  ("rosenbrock", rosen, np.array([-1.2, 1.0]), 3000)]:
        out[oname] = {}
        for n in FIRST:
            finals = []
            for lr in grid:
                _, ls = run(make(n, lr), x0, obj["g"], obj["f"], iters)
                finals.append(float(ls[-1]) if np.isfinite(ls[-1]) and ls[-1] < 1e6 else None)
            out[oname][n] = finals
    p = plt(); fig, axes = p.subplots(1, 2, figsize=(12, 4))
    for ax, oname in zip(axes, out):
        for n in FIRST:
            y = [v if v is not None else np.nan for v in out[oname][n]]
            ax.loglog(grid, np.maximum(y, 1e-16), "o-", ms=3, label=n)
        if oname == "quadratic": ax.axvline(2 / L, color="r", ls=":", label="2/L (GD stability limit)")
        ax.set_xlabel("learning rate"); ax.set_ylabel("final f (missing = diverged)"); ax.set_title(oname); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(outdir(T), "A4_stepsize.png")); p.close(fig)
    out["grid"] = grid
    return out


def a5_noisy(seeds=10, sigma=2.0, iters=2000):
    out = {}
    configs = {"SGD const lr=0.01": lambda k: 0.01, "SGD const lr=0.001": lambda k: 0.001,
               "SGD decay 1/(mu(k+k0))": lambda k: 1.0 / (MU * (k + 50))}
    for name, sched in configs.items():
        finals = []
        for s in range(seeds):
            rng = np.random.default_rng(s); x = np.array([1.0, 1.0])
            for k in range(iters):
                x = x - sched(k) * (quad["g"](x) + sigma * rng.standard_normal(2))
            finals.append(quad["f"](x))
        out[name] = mean_std(finals)
    for name, mk in {"Adam lr=0.01": lambda: Adam(0.01), "Momentum lr=0.002": lambda: Momentum(0.002)}.items():
        finals = []
        for s in range(seeds):
            rng = np.random.default_rng(s); x = np.array([1.0, 1.0]); o = mk()
            for k in range(iters):
                x = o.step(x, quad["g"](x) + sigma * rng.standard_normal(2))
            finals.append(quad["f"](x))
        out[name] = mean_std(finals)
    return {k: {"mean_final_f": v[0], "std": v[1]} for k, v in out.items()}


if __name__ == "__main__":
    seed_all(0)
    r = {"A1_convergence": a1_convergence(), "A2_saddle": a2_saddle(), "A3_local_minima": a3_local_minima(),
         "A4_stepsize": a4_stepsize(), "A5_noisy": a5_noisy()}
    save_json(T, "landscapes.json", r)
    for o, d in r["A1_convergence"].items():
        print(o)
        for n, v in d.items(): print("  ", n, v)
    print("SADDLE", r["A2_saddle"]); print("LOCALMIN", r["A3_local_minima"]); print("NOISY", r["A5_noisy"])
