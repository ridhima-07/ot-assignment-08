"""Track D3: distributed and multi-task optimisation.

  F1 FedAvg vs centralised training (logistic regression, 10 clients, IID and label-skewed
     non-IID splits). Vary local steps E and communication rounds; count rounds and
     floats communicated to reach a target accuracy. 5 seeds each.
  M1 Two-task model (shared trunk + two heads) with conflicting tasks of different scale.
     Equal weighting vs MGDA (min-norm, closed form for 2 tasks) vs GradNorm (alpha = 1.5).
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.datasets import make_classification
from common.util import save_json, outdir, plt, mean_std

T = "trackD"


def sig(z): return 1 / (1 + np.exp(-z))


def data(seed=0):
    X, y = make_classification(4000, 20, n_informative=10, random_state=seed, class_sep=0.8)
    mu, sd = X[:3000].mean(0), X[:3000].std(0); X = (X - mu) / sd          # standardise with train statistics
    X = np.c_[X, np.ones(len(X))]; return X[:3000], y[:3000], X[3000:], y[3000:]


def grad(w, X, y, lam=1e-3): return X.T @ (sig(X @ w) - y) / len(y) + lam * w
def loss(w, X, y, lam=1e-3):
    z = X @ w; return float(np.mean(np.logaddexp(0, z) - y * z) + 0.5 * lam * w @ w)
def acc(w, X, y): return float(np.mean((sig(X @ w) > 0.5) == y))


def split(X, y, K, iid, rng):
    if iid:
        idx = rng.permutation(len(y)); return [(X[i], y[i]) for i in np.array_split(idx, K)]
    # label skew: each client gets 90% of one class
    pos, neg = rng.permutation(np.flatnonzero(y == 1)), rng.permutation(np.flatnonzero(y == 0)); parts = []
    pp, nn = np.array_split(pos, K), np.array_split(neg, K)
    for k in range(K):
        a, b = (pp[k], nn[k]) if True else None
        major, minor = (a, b) if k % 2 == 0 else (b, a)
        keep = minor[: max(1, len(major) // 9)]; parts.append(np.r_[major, keep])
    return [(X[i], y[i]) for i in parts]


def fedavg(clients, rounds, E, lr, bs, rng, Xtr, ytr, fstar, target):
    """target = suboptimality of the GLOBAL training objective, f(w) - f*."""
    d = clients[0][0].shape[1]; w = np.zeros(d); hist = []; reached = None
    sizes = np.array([len(c[1]) for c in clients], float)
    for r in range(rounds):
        locals_ = []
        for Xc, yc in clients:
            wl = w.copy()
            for _ in range(E):
                b = rng.choice(len(yc), bs); wl -= lr * grad(wl, Xc[b], yc[b])
            locals_.append(wl)
        w = np.average(locals_, axis=0, weights=sizes); g = loss(w, Xtr, ytr) - fstar; hist.append(g)
        if reached is None and g <= target: reached = r + 1
    return w, hist, reached


def central(Xtr, ytr, steps, lr, bs, rng, fstar):
    w = np.zeros(Xtr.shape[1])
    for _ in range(steps):
        b = rng.choice(len(ytr), bs); w -= lr * grad(w, Xtr[b], ytr[b])
    return loss(w, Xtr, ytr) - fstar


def f1_fedavg(K=10, rounds=60, seeds=5):
    Xtr, ytr, Xte, yte = data(); out = {}
    from scipy.optimize import minimize
    # reference optimum by L-BFGS (a fixed-step GD reference diverged on unscaled features)
    w_ref = minimize(lambda w: loss(w, Xtr, ytr), np.zeros(Xtr.shape[1]), jac=lambda w: grad(w, Xtr, ytr),
                     method="L-BFGS-B", options={"gtol": 1e-12, "ftol": 1e-16, "maxiter": 10000}).x
    assert np.linalg.norm(grad(w_ref, Xtr, ytr)) < 1e-5, "reference optimum not converged"
    fstar = loss(w_ref, Xtr, ytr); target = 2e-3
    out["centralised_optimum"] = {"f*": fstar, "test_acc": acc(w_ref, Xte, yte)}; out["target_suboptimality"] = target
    out["centralised_minibatch_SGD(same total gradient steps as FedAvg E=5)"] = {
        "final_subopt_mean": mean_std([central(Xtr, ytr, rounds * 5 * K, 0.05, 32, np.random.default_rng(s), fstar) for s in range(seeds)])[0],
        "note": "needs a model exchange every step if run across clients: rounds*E*K messages"}
    curves = {}
    for iid in [True, False]:
        for E in [1, 5, 20, 50]:
            accs, rr = [], []; cs = []
            for s in range(seeds):
                rng = np.random.default_rng(s); cl = split(Xtr, ytr, K, iid, rng)
                w, h, reached = fedavg(cl, rounds, E, 0.05, 32, rng, Xtr, ytr, fstar, target)
                accs.append(h[-1]); rr.append(reached); cs.append(h)
            ok = [r for r in rr if r is not None]
            key = f"{'IID' if iid else 'non-IID'} E={E}"
            out[key] = {"final_subopt_mean": mean_std(accs)[0], "final_subopt_std": mean_std(accs)[1], "test_acc_final": acc(w, Xte, yte),
                        "rounds_to_target_mean": float(np.mean(ok)) if ok else None, "runs_reaching_target": f"{len(ok)}/{seeds}",
                        "floats_communicated_to_target": float(np.mean(ok)) * K * 2 * Xtr.shape[1] if ok else None}
            curves[key] = np.mean(cs, axis=0)
    p = plt(); fig, ax = p.subplots(figsize=(6.5, 4))
    for k, c in curves.items(): ax.semilogy(c, label=k, ls="-" if "non" not in k else "--")
    ax.axhline(target, color="k", ls=":", label="target"); ax.set_xlabel("communication round"); ax.set_ylabel("f(w) - f* (global train loss)")
    ax.legend(fontsize=7); ax.set_title("FedAvg (10 clients, mean of 5 seeds)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir(T), "D_fedavg.png")); p.close(fig)
    return out


# ---------------- multi-task ----------------
def mt_data(seed=0, n=2000):
    # task vectors are FIXED (seed 12345); only the samples X depend on `seed`.
    # (Bug fixed: originally a, b were drawn from the same rng as X, so the validation
    #  set described different tasks than the training set.)
    trng = np.random.default_rng(12345); a = trng.standard_normal(10); b = -0.6 * a + 0.8 * trng.standard_normal(10)
    rng = np.random.default_rng(seed); X = rng.standard_normal((n, 10))
    y1 = np.tanh(X @ a) ; y2 = 10.0 * np.tanh(X @ b)                               # task 2 has 10x scale
    return X, y1, y2


def mt_forward(P, X):
    Hh = np.tanh(X @ P["W"]); return Hh, Hh @ P["v1"], Hh @ P["v2"]


def mt_losses_grads(P, X, y1, y2):
    Hh, p1, p2 = mt_forward(P, X); n = len(X)
    L1 = 0.5 * np.mean((p1 - y1) ** 2); L2 = 0.5 * np.mean((p2 - y2) ** 2)
    d1 = (p1 - y1) / n; d2 = (p2 - y2) / n
    gv1 = Hh.T @ d1; gv2 = Hh.T @ d2
    gW1 = X.T @ (np.outer(d1, P["v1"]) * (1 - Hh ** 2)); gW2 = X.T @ (np.outer(d2, P["v2"]) * (1 - Hh ** 2))
    return L1, L2, gW1, gW2, gv1, gv2


def mt_train(method, seed, steps=3000, lr=0.02, alpha=1.5):
    X, y1, y2 = mt_data(seed); Xv, y1v, y2v = mt_data(seed + 100)
    rng = np.random.default_rng(seed); P = {"W": rng.standard_normal((10, 16)) * 0.3, "v1": np.zeros(16), "v2": np.zeros(16)}
    w = np.ones(2); L0 = None; wl = 0.025
    for t in range(steps):
        L1, L2, gW1, gW2, gv1, gv2 = mt_losses_grads(P, X, y1, y2)
        if L0 is None: L0 = np.array([L1, L2])
        if method == "equal":
            c = np.array([1.0, 1.0])
        elif method == "MGDA":
            g1, g2 = gW1.ravel(), gW2.ravel()               # min-norm point in conv{g1, g2} on the SHARED params
            a = np.clip((g2 - g1) @ g2 / max((g1 - g2) @ (g1 - g2), 1e-12), 0, 1); c = np.array([a, 1 - a]) * 2
        else:  # GradNorm on the shared layer
            G = np.array([np.linalg.norm(w[0] * gW1), np.linalg.norm(w[1] * gW2)]); Gbar = G.mean()
            Lt = np.array([L1, L2]) / L0; rinv = Lt / Lt.mean(); target = Gbar * rinv ** alpha
            # d/dw_i |G_i - target_i|  with target treated as constant (as in the paper)
            gw = np.sign(G - target) * np.array([np.linalg.norm(gW1), np.linalg.norm(gW2)])
            w = np.maximum(w - wl * gw, 1e-3); w = 2 * w / w.sum(); c = w
        P["W"] -= lr * (c[0] * gW1 + c[1] * gW2); P["v1"] -= lr * gv1; P["v2"] -= lr * gv2
    L1v, L2v = mt_losses_grads(P, Xv, y1v, y2v)[:2]
    var1, var2 = 0.5 * np.var(y1v), 0.5 * np.var(y2v)              # loss of predicting the mean
    return {"task1_rel_loss": L1v / var1, "task2_rel_loss": L2v / var2, "final_weights": c.round(3).tolist()}


def m1_multitask(seeds=5):
    out = {}
    for m in ["equal", "MGDA", "GradNorm"]:
        rs = [mt_train(m, s) for s in range(seeds)]
        out[m] = {"task1_rel_val_loss": mean_std([r["task1_rel_loss"] for r in rs]),
                  "task2_rel_val_loss": mean_std([r["task2_rel_loss"] for r in rs]),
                  "mean_rel_loss": float(np.mean([(r["task1_rel_loss"] + r["task2_rel_loss"]) / 2 for r in rs])),
                  "example_final_task_weights": rs[0]["final_weights"]}
    out["note"] = "rel loss = validation MSE / variance of target (1.0 = no better than predicting the mean). (mean, std) over 5 seeds."
    return out


if __name__ == "__main__":
    r = {"F1_fedavg": f1_fedavg(), "M1_multitask": m1_multitask()}
    save_json(T, "distributed.json", r); print(json.dumps(r, indent=1, default=str))
