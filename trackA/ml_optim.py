"""Track A, part 2: optimization inside machine learning.

  A6  ML as optimization: logistic regression = parameters + loss + L2 regulariser (convex)
  A7  GD vs SGD vs mini-batch SGD and all adaptive methods on the same data, 5 seeds each
  A8  Backpropagation (hand-written for a 2-layer MLP), verified by finite differences,
      kept separate from the optimizer that consumes the gradient
  A9  Curvature: exact Hessian, Hessian-vector products (analytic and finite-difference),
      Fisher / Gauss-Newton, diagonal and low-rank (Lanczos) approximations,
      Newton, Hessian-free Newton-CG, natural gradient
  A10 Gauss-Newton and Levenberg-Marquardt on a nonlinear least-squares curve fit
  A11 Continuous vs discrete: L1 relaxation vs exhaustive best-subset selection
"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
np.seterr(all="ignore")
from scipy.sparse.linalg import LinearOperator, eigsh, cg
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso
from common.util import save_json, outdir, plt, timer, mean_std
from trackA.optimizers import GD, Momentum, NAG, AdaGrad, RMSProp, Adam, GaussNewton

T = "trackA"

# ---------------- data ----------------
Xall, yall = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xall, yall, test_size=0.3, random_state=0, stratify=yall)
sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))]); Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
N, D = Xtr.shape
LAM = 1e-2


def sig(z): return 1 / (1 + np.exp(-z))


def logreg_loss(w, X=Xtr, y=ytr, lam=LAM):
    z = X @ w
    return float(np.mean(np.logaddexp(0, z) - y * z) + 0.5 * lam * w[:-1] @ w[:-1])


def logreg_grad(w, X=Xtr, y=ytr, lam=LAM):
    r = lam * w; r[-1] = 0
    return X.T @ (sig(X @ w) - y) / len(y) + r


def logreg_hess(w, X=Xtr, lam=LAM):
    s = sig(X @ w); Dm = s * (1 - s)
    reg = lam * np.eye(len(w)); reg[-1, -1] = 0
    return (X.T * Dm) @ X / len(X) + reg


def logreg_hvp(w, v, X=Xtr, lam=LAM):
    """Pearlmutter-style exact HVP without forming H: X^T (D (X v)) / n + lam v."""
    s = sig(X @ w); r = lam * v.copy(); r[-1] = 0
    return X.T @ (s * (1 - s) * (X @ v)) / len(X) + r


def acc(w, X, y): return float(np.mean((sig(X @ w) > 0.5) == y))


# reference optimum (sklearn, same objective: C = 1/(n*lam), bias unpenalised)
ref = LogisticRegression(C=1 / (N * LAM), tol=1e-12, max_iter=10000).fit(Xtr[:, :-1], ytr)
w_ref = np.r_[ref.coef_.ravel(), ref.intercept_]
F_STAR = logreg_loss(w_ref)


def a6_regularisation():
    out = {}
    for lam in [0.0, 1e-3, 1e-2, 1e-1, 1.0]:
        w = np.zeros(D)
        for _ in range(3000):
            w = w - 0.5 * logreg_grad(w, lam=lam)
        out[str(lam)] = {"train_loss": logreg_loss(w, lam=lam), "||w||": float(np.linalg.norm(w[:-1])),
                         "train_acc": acc(w, Xtr, ytr), "test_acc": acc(w, Xte, yte)}
    return out


def a7_optimizers(epochs=30, seeds=5, batch=32):
    """Same epoch budget (gradient evaluations per sample) for every method."""
    lrs = {"GD (full batch)": 1.0, "SGD (batch=1)": 0.05, "Mini-batch SGD": 0.3, "Momentum": 0.1,
           "NAG": 0.1, "AdaGrad": 0.3, "RMSProp": 0.01, "Adam": 0.01}
    mk = {"GD (full batch)": lambda lr: GD(lr), "SGD (batch=1)": lambda lr: GD(lr), "Mini-batch SGD": lambda lr: GD(lr),
          "Momentum": lambda lr: Momentum(lr), "NAG": lambda lr: NAG(lr), "AdaGrad": lambda lr: AdaGrad(lr),
          "RMSProp": lambda lr: RMSProp(lr), "Adam": lambda lr: Adam(lr)}
    out = {}; curves = {}
    for name, lr in lrs.items():
        finals, accs, times, cs = [], [], [], []
        for s in range(seeds):
            rng = np.random.default_rng(s); w = np.zeros(D); opt = mk[name](lr); curve = [logreg_loss(w) - F_STAR]
            bs = N if name.startswith("GD") else (1 if name.startswith("SGD") else batch)
            with timer() as t:
                for ep in range(epochs):
                    idx = rng.permutation(N)
                    for i in range(0, N, bs):
                        b = idx[i:i + bs]
                        if getattr(opt, "needs_lookahead", False):
                            g = logreg_grad(opt.lookahead(w), Xtr[b], ytr[b])
                        else:
                            g = logreg_grad(w, Xtr[b], ytr[b])
                        w = opt.step(w, g)
                    curve.append(logreg_loss(w) - F_STAR)
            finals.append(curve[-1]); accs.append(acc(w, Xte, yte)); times.append(t["s"]); cs.append(curve)
        m, sd = mean_std(finals)
        out[name] = {"lr": lr, "batch": bs, "subopt_mean": m, "subopt_std": sd,
                     "test_acc_mean": mean_std(accs)[0], "secs_mean": mean_std(times)[0]}
        curves[name] = np.mean(cs, axis=0)
    p = plt(); fig, ax = p.subplots(figsize=(7, 4.5))
    for n, c in curves.items(): ax.semilogy(np.maximum(c, 1e-12), label=n)
    ax.set_xlabel("epoch"); ax.set_ylabel("f(w) - f* (mean of 5 seeds)"); ax.set_title("Logistic regression, breast-cancer data")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(outdir(T), "A7_ml_optimizers.png")); p.close(fig)
    out["f_star_sklearn"] = F_STAR
    return out


# ---------------- A8: backprop for a 2-layer MLP ----------------
class MLP:
    """x -> tanh(W1 x + b1) -> W2 h + b2 -> sigmoid; binary cross-entropy loss."""
    def __init__(self, din, h, rng):
        self.shapes = [(h, din), (h,), (1, h), (1,)]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.theta = np.concatenate([rng.standard_normal(self.sizes[0]) / np.sqrt(din), np.zeros(h),
                                     rng.standard_normal(self.sizes[2]) / np.sqrt(h), np.zeros(1)])

    def unpack(self, th):
        out, i = [], 0
        for s, n in zip(self.shapes, self.sizes): out.append(th[i:i + n].reshape(s)); i += n
        return out

    def forward(self, th, X):
        W1, b1, W2, b2 = self.unpack(th)
        a1 = X @ W1.T + b1; h = np.tanh(a1); z = (h @ W2.T + b2).ravel()
        return z, (X, a1, h)

    def loss(self, th, X, y):
        z, _ = self.forward(th, X)
        return float(np.mean(np.logaddexp(0, z) - y * z))

    def backprop(self, th, X, y):
        """Reverse-mode chain rule, layer by layer. Returns dL/dtheta (no optimizer here)."""
        W1, b1, W2, b2 = self.unpack(th)
        z, (X, a1, h) = self.forward(th, X)
        dz = (sig(z) - y)[:, None] / len(y)          # dL/dz
        dW2 = dz.T @ h; db2 = dz.sum(0)               # output layer
        dh = dz @ W2                                  # back through W2
        da1 = dh * (1 - h ** 2)                       # back through tanh
        dW1 = da1.T @ X; db1 = da1.sum(0)
        return np.concatenate([dW1.ravel(), db1, dW2.ravel(), db2])


def a8_backprop():
    rng = np.random.default_rng(0)
    X, y = make_moons(400, noise=0.2, random_state=0)
    Xm, Xv, ym, yv = train_test_split(X, y, test_size=0.3, random_state=0)
    net = MLP(2, 16, rng); th = net.theta.copy()
    # gradient check: central finite differences on every parameter
    g = net.backprop(th, Xm, ym); fd = np.zeros_like(th); e = 1e-6
    for i in range(len(th)):
        d = np.zeros_like(th); d[i] = e
        fd[i] = (net.loss(th + d, Xm, ym) - net.loss(th - d, Xm, ym)) / (2 * e)
    rel_err = float(np.linalg.norm(g - fd) / np.linalg.norm(g + fd))
    out = {"n_params": len(th), "gradcheck_rel_error": rel_err}
    # same backprop gradient fed to different optimizers (separation of concerns)
    for name, opt in [("GD", GD(0.5)), ("Momentum", Momentum(0.1)), ("Adam", Adam(0.02)), ("RMSProp", RMSProp(0.01))]:
        accs = []
        for s in range(5):
            th = MLP(2, 16, np.random.default_rng(s)).theta; o = type(opt)(opt.lr)
            for _ in range(2000): th = o.step(th, net.backprop(th, Xm, ym))
            z, _ = net.forward(th, Xv); accs.append(float(np.mean((z > 0) == yv)))
        out[name] = {"val_acc_mean": mean_std(accs)[0], "val_acc_std": mean_std(accs)[1]}
    # landscape: two different seeds reach different parameter vectors with ~equal loss (non-convexity)
    ths = []
    for s in range(2):
        th = MLP(2, 16, np.random.default_rng(100 + s)).theta; o = Adam(0.02)
        for _ in range(3000): th = o.step(th, net.backprop(th, Xm, ym))
        ths.append(th)
    alphas = np.linspace(-0.25, 1.25, 31)
    path = [net.loss((1 - a) * ths[0] + a * ths[1], Xm, ym) for a in alphas]
    out["interpolation_barrier"] = float(max(path) - max(path[10], path[20]))
    p = plt(); fig, ax = p.subplots(figsize=(6, 4))
    ax.plot(alphas, path, "o-"); ax.set_xlabel("alpha (0 = solution 1, 1 = solution 2)"); ax.set_ylabel("train loss")
    ax.set_title("MLP loss along the line between two minima (non-convex)"); fig.tight_layout()
    fig.savefig(os.path.join(outdir(T), "A8_mlp_interpolation.png")); p.close(fig)
    return out


# ---------------- A9: curvature ----------------
def a9_curvature():
    out = {}
    # NOTE: evaluate curvature at a non-trivial point. At w=0 every sigmoid is 0.5, so
    # (s-y)^2 = s(1-s) = 0.25 and the empirical Fisher coincides with the Hessian by accident
    # (this misleading result is recorded in docs/FAILURE_LOG.md).
    w = 0.5 * w_ref
    H = logreg_hess(w); eig = np.linalg.eigvalsh(H)
    out["hessian_eigs"] = {"min": float(eig.min()), "max": float(eig.max()), "condition": float(eig.max() / eig.min())}
    # HVP checks
    v = np.random.default_rng(0).standard_normal(D)
    hv_exact = H @ v; hv_pearl = logreg_hvp(w, v)
    hv_fd = (logreg_grad(w + 1e-5 * v) - logreg_grad(w - 1e-5 * v)) / 2e-5
    out["hvp_rel_err_pearlmutter"] = float(np.linalg.norm(hv_pearl - hv_exact) / np.linalg.norm(hv_exact))
    out["hvp_rel_err_finite_diff"] = float(np.linalg.norm(hv_fd - hv_exact) / np.linalg.norm(hv_exact))
    # Fisher: for logistic regression the model Fisher equals the GGN equals the Hessian (data part)
    s = sig(Xtr @ w); F_model = (Xtr.T * (s * (1 - s))) @ Xtr / N
    gi = Xtr * (s - ytr)[:, None]; F_emp = gi.T @ gi / N        # empirical Fisher
    Hd = H - np.diag(np.r_[np.full(D - 1, LAM), 0])
    out["model_fisher_vs_hessian_rel_diff"] = float(np.linalg.norm(F_model - Hd) / np.linalg.norm(Hd))
    out["empirical_fisher_vs_hessian_rel_diff"] = float(np.linalg.norm(F_emp - Hd) / np.linalg.norm(Hd))
    # low-rank: top-k eigenpairs from HVPs only (Lanczos)
    op = LinearOperator((D, D), matvec=lambda u: logreg_hvp(w, u))
    vals, _ = eigsh(op, k=5, which="LA")
    out["lanczos_top5"] = sorted(vals.tolist(), reverse=True)
    out["exact_top5"] = sorted(eig[-5:].tolist(), reverse=True)

    # optimisation comparison: iterations and HVP/gradient counts to reach 1e-8 suboptimality
    def until(update, maxit=2000, tol=1e-8):
        w = np.zeros(D)
        for k in range(1, maxit + 1):
            w = update(w)
            if logreg_loss(w) - F_STAR < tol: return {"iters": k, "final_subopt": float(logreg_loss(w) - F_STAR)}
        return {"iters": None, "final_subopt": float(logreg_loss(w) - F_STAR), "budget": maxit}
    Lc = eig.max()
    res = {}
    Lc = max(Lc, np.linalg.eigvalsh(logreg_hess(np.zeros(D))).max())
    res["GD (lr=1/L)"] = until(lambda w: w - logreg_grad(w) / Lc)
    # diagonal approximation ignores feature correlations; step must be tuned (sweep, report all)
    diag_sweep = {str(a): until(lambda w, a=a: w - a * logreg_grad(w) / np.diag(logreg_hess(w))) for a in [0.5, 0.2, 0.1, 0.05, 0.02]}
    out["diag_precond_step_sweep"] = diag_sweep
    res["Diagonal-Hessian preconditioned GD"] = min(diag_sweep.values(), key=lambda d: (d["iters"] is None, d["iters"] or 0, d["final_subopt"]))
    res["Newton (exact H)"] = until(lambda w: w - np.linalg.solve(logreg_hess(w), logreg_grad(w)))
    reg = LAM * np.eye(D); reg[-1, -1] = 1e-6
    def nat(w):   # Fisher of the likelihood + Fisher of the Gaussian prior (= L2 term)
        s = sig(Xtr @ w); F = (Xtr.T * (s * (1 - s))) @ Xtr / N + reg
        return w - np.linalg.solve(F, logreg_grad(w))
    res["Natural gradient (model Fisher)"] = until(nat)
    def empf(w):
        s = sig(Xtr @ w); gi = Xtr * (s - ytr)[:, None]; F = gi.T @ gi / N + reg
        return w - 0.5 * np.linalg.solve(F, logreg_grad(w))
    res["Empirical-Fisher preconditioning"] = until(empf, maxit=500)
    hvps = {"n": 0}
    def hf(w):
        def mv(u): hvps["n"] += 1; return logreg_hvp(w, u)
        d, _ = cg(LinearOperator((D, D), matvec=mv), logreg_grad(w), maxiter=20)
        return w - d
    res["Hessian-free Newton-CG"] = until(hf); out["newton_cg_total_hvps"] = hvps["n"]
    def lowrank(w, k=5):
        Hm = logreg_hess(w); vals, V = np.linalg.eigh(Hm); V, vals = V[:, -k:], vals[-k:]
        g = logreg_grad(w); gp = V.T @ g
        return w - (V @ (gp / vals)) - (g - V @ gp) / vals[0]   # Newton in top-k subspace, scaled GD elsewhere
    res["Low-rank (top-5) Newton"] = until(lowrank)
    out["iterations_to_1e-8"] = res
    return out


# ---------------- A10: Gauss-Newton ----------------
def a10_gauss_newton():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 4, 60); true = np.array([2.5, 1.3, 0.5])
    yobs = true[0] * np.exp(-true[1] * t) + true[2] + 0.02 * rng.standard_normal(len(t))
    def r(p): return p[0] * np.exp(-p[1] * t) + p[2] - yobs
    def J(p): return np.stack([np.exp(-p[1] * t), -p[0] * t * np.exp(-p[1] * t), np.ones_like(t)], 1)
    f = lambda p: 0.5 * r(p) @ r(p)
    out = {}
    for name, step in [("Gauss-Newton", GaussNewton(0.0)), ("Levenberg-Marquardt (lam=1e-2)", GaussNewton(1e-2))]:
        p = np.array([1.0, 0.5, 0.0]); hist = [f(p)]
        for _ in range(30):
            p = step.step(p, J=J(p), r=r(p)); hist.append(f(p))
        out[name] = {"params": p.round(4).tolist(), "final_loss": hist[-1], "iters_to_converge": int(np.argmax(np.abs(np.diff(hist)) < 1e-12))}
    fgn = out["Gauss-Newton"]["final_loss"]
    p = np.array([1.0, 0.5, 0.0]); o = Adam(0.05); k_adam = None
    for k in range(20000):
        p = o.step(p, J(p).T @ r(p))
        if k_adam is None and f(p) - fgn < 1e-9: k_adam = k + 1
    out["Adam"] = {"params": p.round(4).tolist(), "final_loss": f(p), "iters_to_within_1e-9_of_GN": k_adam}
    p = np.array([1.0, 0.5, 0.0]); k_gd = None
    for k in range(20000):
        p = p - 0.01 * J(p).T @ r(p)
        if k_gd is None and f(p) - fgn < 1e-9: k_gd = k + 1
    out["GD (lr=0.01)"] = {"params": p.round(4).tolist(), "final_loss": f(p), "iters_to_within_1e-9_of_GN": k_gd}
    out["true_params"] = true.tolist()
    return out


# ---------------- A11: continuous vs discrete ----------------
def a11_continuous_vs_discrete(k=4, nfeat=12):
    """Pick k of the first 12 features. Discrete: exhaustive over C(12,4)=495 subsets.
    Continuous: L1 (Lasso) relaxation, tune alpha until k non-zeros."""
    X = Xtr[:, :nfeat]; y = ytr.astype(float)
    def sse(S):
        A = np.c_[X[:, list(S)], np.ones(N)]; coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return float(np.sum((A @ coef - y) ** 2))
    with timer() as t1:
        best = min(itertools.combinations(range(nfeat), k), key=sse)
    with timer() as t2:
        for a in np.logspace(-3, 0, 200):
            nz = np.flatnonzero(Lasso(alpha=a, max_iter=10000).fit(X, y).coef_)
            if len(nz) <= k: break
    return {"exhaustive_subset": list(best), "exhaustive_sse": sse(best), "exhaustive_secs": t1["s"],
            "lasso_subset": nz.tolist(), "lasso_sse_refit": sse(nz), "lasso_secs": t2["s"],
            "lasso_alpha": float(a), "subsets_enumerated": 495}


if __name__ == "__main__":
    r = {"A6_regularisation": a6_regularisation(), "A7_optimizers": a7_optimizers(), "A8_backprop": a8_backprop(),
         "A9_curvature": a9_curvature(), "A10_gauss_newton": a10_gauss_newton(),
         "A11_continuous_vs_discrete": a11_continuous_vs_discrete()}
    save_json(T, "ml_optim.json", r)
    import json; print(json.dumps(r, indent=1, default=str))
