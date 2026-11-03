"""First- and second-order optimizers implemented from scratch (NumPy).

Every optimizer has the same interface:
    step(x, g, **ctx) -> new x
where g is a gradient (possibly stochastic). Second-order methods also receive
a Hessian (or a Hessian-vector-product function) through ctx.
"""
import numpy as np


class GD:
    name = "GD"
    def __init__(self, lr): self.lr = lr
    def step(self, x, g, **_): return x - self.lr * g


class Momentum:
    """Heavy-ball momentum: v <- beta v + g ; x <- x - lr v."""
    name = "Momentum"
    def __init__(self, lr, beta=0.9): self.lr, self.beta, self.v = lr, beta, None
    def step(self, x, g, **_):
        self.v = g if self.v is None else self.beta * self.v + g
        return x - self.lr * self.v


class NAG:
    """Nesterov accelerated gradient, look-ahead form.
    Needs the gradient at the look-ahead point: caller supplies grad_fn in ctx."""
    name = "NAG"
    needs_lookahead = True
    def __init__(self, lr, beta=0.9): self.lr, self.beta, self.v = lr, beta, None
    def lookahead(self, x):
        return x if self.v is None else x - self.lr * self.beta * self.v
    def step(self, x, g_look, **_):
        self.v = g_look if self.v is None else self.beta * self.v + g_look
        return x - self.lr * self.v


class AdaGrad:
    name = "AdaGrad"
    def __init__(self, lr, eps=1e-8): self.lr, self.eps, self.G = lr, eps, None
    def step(self, x, g, **_):
        self.G = g * g if self.G is None else self.G + g * g
        return x - self.lr * g / (np.sqrt(self.G) + self.eps)


class RMSProp:
    name = "RMSProp"
    def __init__(self, lr, rho=0.9, eps=1e-8): self.lr, self.rho, self.eps, self.s = lr, rho, eps, None
    def step(self, x, g, **_):
        self.s = (1 - self.rho) * g * g if self.s is None else self.rho * self.s + (1 - self.rho) * g * g
        return x - self.lr * g / (np.sqrt(self.s) + self.eps)


class Adam:
    name = "Adam"
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = self.v = None; self.t = 0
    def step(self, x, g, **_):
        self.t += 1
        if self.m is None: self.m, self.v = np.zeros_like(g), np.zeros_like(g)
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t); vh = self.v / (1 - self.b2 ** self.t)
        return x - self.lr * mh / (np.sqrt(vh) + self.eps)


class Newton:
    """Damped Newton: x <- x - lr (H + damping I)^-1 g. Needs H in ctx."""
    name = "Newton"
    def __init__(self, lr=1.0, damping=0.0): self.lr, self.damping = lr, damping
    def step(self, x, g, H=None, **_):
        Hd = H + self.damping * np.eye(len(x))
        return x - self.lr * np.linalg.solve(Hd, g)


class GaussNewton:
    """For least squares f = 0.5||r(x)||^2: x <- x - (J^T J + lam I)^-1 J^T r.
    lam=0 is pure Gauss-Newton; lam>0 is Levenberg-Marquardt style damping."""
    name = "Gauss-Newton"
    def __init__(self, lam=0.0): self.lam = lam
    def step(self, x, g=None, J=None, r=None, **_):
        A = J.T @ J + self.lam * np.eye(J.shape[1])
        return x - np.linalg.solve(A, J.T @ r)


def run(opt, x0, grad_fn, f_fn, iters, hess_fn=None, record_every=1):
    """Generic driver for deterministic objectives; returns trajectory and losses."""
    x = np.array(x0, dtype=float)
    traj, losses = [x.copy()], [f_fn(x)]
    for k in range(iters):
        if getattr(opt, "needs_lookahead", False):
            g = grad_fn(opt.lookahead(x))
        else:
            g = grad_fn(x)
        ctx = {"H": hess_fn(x)} if hess_fn is not None else {}
        x = opt.step(x, g, **ctx)
        if not np.all(np.isfinite(x)):
            losses.append(np.inf); break
        if (k + 1) % record_every == 0:
            traj.append(x.copy()); losses.append(f_fn(x))
    return np.array(traj), np.array(losses)
