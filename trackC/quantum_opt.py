"""Track C: quantum optimisation on simulators (Qiskit 2.x + Aer). No real hardware was used.

  C1 QAOA (p = 1, 2, 3) on the Track B Max-Cut instance, optimised with COBYLA,
     parameter-shift gradient descent and SPSA; compared with the classical baselines.
  C2 Finite-shot cost: estimator variance vs shots, and QAOA trained with shot noise.
  C3 Hardware noise: depolarising + readout noise model, approximation ratio vs noise level.
  C4 VQE for the H2 molecule (2-qubit reduced Hamiltonian, 0.735 A) with a hardware-efficient
     PQC: parameter-shift GD vs SPSA vs quantum natural gradient vs COBYLA; exact diagonalisation baseline.
  C5 Barren plateaus: variance of a parameter-shift gradient vs number of qubits and depth,
     global vs local cost.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from common.util import save_json, outdir, plt, timer, mean_std, RESULTS

T = "trackC"
inst = json.load(open(os.path.join(RESULTS, "trackB", "maxcut_instance.json")))
N, EDGES, OPT = inst["n"], [tuple(e) for e in inst["edges"]], inst["optimum"]


# ---------------- QAOA ----------------
def cost_op():
    """C = sum_(u,v) (1 - Z_u Z_v)/2 ; we minimise -C."""
    terms = []
    for u, v in EDGES:
        z = ["I"] * N; z[N - 1 - u] = "Z"; z[N - 1 - v] = "Z"; terms.append(("".join(z), 0.5))
    return SparsePauliOp.from_list(terms) + SparsePauliOp.from_list([("I" * N, -0.5 * len(EDGES))])  # = -C


HC = cost_op()


def qaoa_circuit(p, measure=False):
    g, b = ParameterVector("g", p), ParameterVector("b", p)
    qc = QuantumCircuit(N); qc.h(range(N))
    for k in range(p):
        for u, v in EDGES: qc.rzz(-g[k], u, v)      # exp(i g Z_u Z_v / 2) ~ exp(-i g (-C)) up to phase
        for q in range(N): qc.rx(2 * b[k], q)
    if measure: qc.measure_all()
    return qc, list(g) + list(b)


def cut_of_bitstring(bs):   # qiskit bit order: rightmost char is qubit 0
    x = [int(c) for c in bs[::-1]]
    return sum(x[u] != x[v] for u, v in EDGES)


class QAOA:
    def __init__(self, p, shots=None, noise=None, seed=0):
        self.p, self.shots, self.seed = p, shots, seed
        self.qc, self.params = qaoa_circuit(p)
        self.evals = 0
        if shots is not None:
            self.mqc, mparams = qaoa_circuit(p, measure=True)
            self.sim = AerSimulator(noise_model=noise, seed_simulator=seed)
            self.tqc = transpile(self.mqc, self.sim, optimization_level=0)
            self.mparams = mparams

    def energy(self, theta):
        """<-C>. Exact (statevector) if shots is None, else a shot estimate."""
        self.evals += 1
        if self.shots is None:
            sv = Statevector(self.qc.assign_parameters(dict(zip(self.params, theta))))
            return float(np.real(sv.expectation_value(HC)))
        c = self.tqc.assign_parameters(dict(zip(self.tqc.parameters, self._order(theta))))
        counts = self.sim.run(c, shots=self.shots).result().get_counts()
        return -sum(cut_of_bitstring(b) * n for b, n in counts.items()) / self.shots

    def _order(self, theta):
        # bind by NAME: the measured circuit was built with its own ParameterVector objects
        d = {p.name: v for p, v in zip(self.params, theta)}; return [d[p.name] for p in self.tqc.parameters]

    def distribution(self, theta, noise=None, shots=4096):
        qc, ps = qaoa_circuit(self.p, measure=True); sim = AerSimulator(noise_model=noise, seed_simulator=1)
        t = transpile(qc.assign_parameters(dict(zip(ps, theta))), sim, optimization_level=0)
        return sim.run(t, shots=shots).result().get_counts()

    def grad_param_shift(self, theta):
        """Exact for gates exp(-i theta P/2): d<H>/dtheta = [E(theta+pi/2) - E(theta-pi/2)]/2.
        gamma enters each RZZ with coefficient -1 and beta enters RX with coefficient 2,
        so apply the chain rule to the shared parameters."""
        g = np.zeros_like(theta); p = self.p
        # gamma_k appears in |E| RZZ gates with angle -gamma -> shift each gate separately
        for k in range(p):
            gk = 0.0
            for e in range(len(EDGES)):
                gk += -0.5 * (self._energy_gate_shift(theta, k, e, +np.pi / 2) - self._energy_gate_shift(theta, k, e, -np.pi / 2))
            g[k] = gk
            bk = 0.0
            for q in range(N):
                bk += 2 * 0.5 * (self._energy_rx_shift(theta, k, q, +np.pi / 2) - self._energy_rx_shift(theta, k, q, -np.pi / 2))
            g[p + k] = bk
        return g

    def _build_shifted(self, theta, layer, kind, idx, s):
        p = self.p; gam, bet = theta[:p], theta[p:]
        qc = QuantumCircuit(N); qc.h(range(N))
        for k in range(p):
            for e, (u, v) in enumerate(EDGES):
                qc.rzz(-gam[k] + (s if (kind == "zz" and k == layer and e == idx) else 0), u, v)
            for q in range(N):
                qc.rx(2 * bet[k] + (s if (kind == "x" and k == layer and q == idx) else 0), q)
        return qc

    def _energy_gate_shift(self, theta, k, e, s):
        self.evals += 1
        return float(np.real(Statevector(self._build_shifted(theta, k, "zz", e, s)).expectation_value(HC)))

    def _energy_rx_shift(self, theta, k, q, s):
        self.evals += 1
        return float(np.real(Statevector(self._build_shifted(theta, k, "x", q, s)).expectation_value(HC)))


def spsa(f, x0, iters, rng, a=0.2, c=0.1, A=10, alpha=0.602, gamma=0.101):
    x = np.array(x0, float); hist = []
    for k in range(iters):
        ak = a / (k + 1 + A) ** alpha; ck = c / (k + 1) ** gamma
        d = rng.choice([-1, 1], size=len(x))
        gk = (f(x + ck * d) - f(x - ck * d)) / (2 * ck) * d
        x = x - ak * gk; hist.append(f(x))
    return x, hist


def ratio_and_popt(q, theta, noise=None):
    counts = q.distribution(theta, noise)
    tot = sum(counts.values())
    exp_cut = sum(cut_of_bitstring(b) * n for b, n in counts.items()) / tot
    p_opt = sum(n for b, n in counts.items() if cut_of_bitstring(b) == OPT) / tot
    best = max(cut_of_bitstring(b) for b in counts)
    return exp_cut / OPT, p_opt, best


def c1_qaoa(seeds=3):
    out = {}
    for p in [1, 2, 3]:
        out[f"p={p}"] = {}
        for method in ["COBYLA", "param-shift GD", "SPSA"]:
            rs, pops, evs, ts = [], [], [], []
            for s in range(seeds):
                rng = np.random.default_rng(s); x0 = rng.uniform(0, np.pi / 2, 2 * p); q = QAOA(p)
                with timer() as t:
                    if method == "COBYLA":
                        x = minimize(q.energy, x0, method="COBYLA", options={"maxiter": 200}).x
                    elif method == "param-shift GD":
                        x = x0.copy()
                        for _ in range(40): x = x - 0.05 * q.grad_param_shift(x)
                    else:
                        x, _ = spsa(q.energy, x0, 150, rng)
                r, po, best = ratio_and_popt(q, x)
                rs.append(r); pops.append(po); evs.append(q.evals); ts.append(t["s"])
            out[f"p={p}"][method] = {"approx_ratio_mean": mean_std(rs)[0], "approx_ratio_std": mean_std(rs)[1],
                                     "P(optimal cut)_mean": mean_std(pops)[0], "circuit_evals_mean": float(np.mean(evs)),
                                     "secs_mean": float(np.mean(ts))}
    # verify parameter-shift against finite differences
    q = QAOA(2); th = np.array([0.4, 0.7, 0.3, 0.2]); ps = q.grad_param_shift(th)
    fd = np.array([(q.energy(th + 1e-5 * e) - q.energy(th - 1e-5 * e)) / 2e-5 for e in np.eye(4)])
    out["param_shift_vs_fd_max_abs_err"] = float(np.max(np.abs(ps - fd)))
    out["classical_baselines (Track B)"] = {"optimum": OPT, "random_assignment_ratio": 9.0 / OPT,
                                            "simulated_annealing_ratio": 1.0, "greedy_local_search_ratio": 13.15 / OPT}
    return out


def c2_shots():
    q = QAOA(2); th = minimize(q.energy, np.array([0.4, 0.7, 0.3, 0.2]), method="COBYLA").x
    exact = q.energy(th); out = {"exact_energy": exact, "by_shots": {}}
    for shots in [16, 64, 256, 1024, 4096]:
        est = [QAOA(2, shots=shots, seed=s).energy(th) for s in range(30)]
        out["by_shots"][shots] = {"bias": float(np.mean(est) - exact), "std": float(np.std(est, ddof=1)),
                                  "std_x_sqrt_shots": float(np.std(est, ddof=1) * np.sqrt(shots))}
    # optimisation under shot noise (COBYLA on sampled energies), 3 seeds
    out["train_with_shots"] = {}
    for shots in [64, 1024]:
        rs = []
        for s in range(3):
            qs = QAOA(2, shots=shots, seed=s)
            x = minimize(qs.energy, np.random.default_rng(s).uniform(0, np.pi / 2, 4), method="COBYLA", options={"maxiter": 120}).x
            rs.append(ratio_and_popt(QAOA(2), x)[0])
        out["train_with_shots"][shots] = {"approx_ratio_mean": mean_std(rs)[0], "std": mean_std(rs)[1]}
    p = plt(); fig, ax = p.subplots(figsize=(5.5, 4))
    sh = list(out["by_shots"]); ax.loglog(sh, [out["by_shots"][s]["std"] for s in sh], "o-", label="measured std")
    ax.loglog(sh, [out["by_shots"][16]["std"] * np.sqrt(16 / s) for s in sh], "k--", label="1/sqrt(shots)")
    ax.set_xlabel("shots"); ax.set_ylabel("std of energy estimate"); ax.legend(); ax.set_title("Finite-shot cost (QAOA p=2)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir(T), "C2_shots.png")); p.close(fig)
    return out, th


def noise_model(p1, p2, ro):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["rx", "h"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["rzz"])
    if ro > 0: nm.add_all_qubit_readout_error(ReadoutError([[1 - ro, ro], [ro, 1 - ro]]))
    return nm


def c3_noise(th):
    q = QAOA(2); out = {}
    for name, (p1, p2, ro) in {"noiseless": (0, 0, 0), "low (1e-4,1e-3,1%)": (1e-4, 1e-3, 0.01),
                                "medium (1e-3,1e-2,2%)": (1e-3, 1e-2, 0.02), "high (5e-3,5e-2,5%)": (5e-3, 5e-2, 0.05)}.items():
        nm = None if p1 == p2 == ro == 0 else noise_model(p1, p2, ro)
        r, po, best = ratio_and_popt(q, th, nm)
        out[name] = {"approx_ratio": r, "P(optimal)": po, "best_sampled_cut": best}
    out["random_guess_ratio"] = 9.0 / OPT
    return out


# ---------------- VQE ----------------
H2 = SparsePauliOp.from_list([("II", -1.052373245772859), ("IZ", 0.39793742484318045), ("ZI", -0.39793742484318045),
                              ("ZZ", -0.01128010425623538), ("XX", 0.18093119978423156)])
E_EXACT = float(np.linalg.eigvalsh(H2.to_matrix()).min())


def vqe_ansatz(layers=2):
    th = ParameterVector("t", 4 * layers); qc = QuantumCircuit(2); qc.x(0)   # Hartree-Fock reference |01>
    i = 0
    for _ in range(layers):
        qc.ry(th[i], 0); qc.ry(th[i + 1], 1); qc.cx(0, 1); qc.rz(th[i + 2], 0); qc.rz(th[i + 3], 1); i += 4
    return qc, list(th)


VQC, VP = vqe_ansatz()


def vqe_state(t): return Statevector(VQC.assign_parameters(dict(zip(VP, t))))
def vqe_E(t): return float(np.real(vqe_state(t).expectation_value(H2)))


def vqe_grad(t):   # parameter shift: every parameter sits in exactly one Pauli rotation
    return np.array([(vqe_E(t + np.pi / 2 * e) - vqe_E(t - np.pi / 2 * e)) / 2 for e in np.eye(len(t))])


def qfi_metric(t, eps=1e-4):
    """Fubini-Study metric g_ij = Re(<d_i psi|d_j psi> - <d_i psi|psi><psi|d_j psi>) (statevector only)."""
    psi = vqe_state(t).data; d = []
    for e in np.eye(len(t)):
        d.append((vqe_state(t + eps * e).data - vqe_state(t - eps * e).data) / (2 * eps))
    d = np.array(d); G = np.real(d.conj() @ d.T - np.outer(d.conj() @ psi, psi.conj() @ d.T))
    return G


def c4_vqe(seeds=5, iters=100):
    out = {"exact_ground_energy": E_EXACT}
    curves = {}
    for method in ["param-shift GD", "SPSA", "QNG", "COBYLA"]:
        errs, its = [], []; cs = []
        for s in range(seeds):
            rng = np.random.default_rng(s); t = rng.uniform(-0.3, 0.3, len(VP)); hist = []
            if method == "param-shift GD":
                for _ in range(iters): t = t - 0.2 * vqe_grad(t); hist.append(vqe_E(t))
            elif method == "QNG":
                for _ in range(iters):
                    G = qfi_metric(t) + 1e-3 * np.eye(len(t)); t = t - 0.2 * np.linalg.solve(G, vqe_grad(t)); hist.append(vqe_E(t))
            elif method == "SPSA":
                t, hist = spsa(vqe_E, t, iters, rng, a=0.3, c=0.1)
            else:
                r = minimize(vqe_E, t, method="COBYLA", options={"maxiter": iters * 4}, callback=lambda xk: hist.append(vqe_E(xk)))
                t = r.x
            e = np.array(hist) - E_EXACT
            errs.append(float(e[-1])); its.append(int(np.argmax(e < 1.6e-3)) + 1 if np.any(e < 1.6e-3) else None); cs.append(e[:iters])
        ok = [i for i in its if i is not None]
        out[method] = {"final_error_Ha_mean": mean_std(errs)[0], "final_error_std": mean_std(errs)[1],
                       "iters_to_chemical_accuracy_mean": float(np.mean(ok)) if ok else None,
                       "runs_reaching_chem_acc": f"{len(ok)}/{seeds}"}
        L = min(len(c) for c in cs); curves[method] = np.mean([c[:L] for c in cs], axis=0)
    p = plt(); fig, ax = p.subplots(figsize=(6, 4))
    for m, c in curves.items(): ax.semilogy(np.maximum(c, 1e-10), label=m)
    ax.axhline(1.6e-3, color="r", ls=":", label="chemical accuracy"); ax.set_xlabel("iteration / callback")
    ax.set_ylabel("E - E_exact (Ha)"); ax.legend(); ax.set_title("VQE for H2 (statevector)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir(T), "C4_vqe.png")); p.close(fig)
    return out


# ---------------- barren plateaus ----------------
def hea(n, layers, rng):
    qc = QuantumCircuit(n); th = rng.uniform(0, 2 * np.pi, n * layers); i = 0
    for _ in range(layers):
        for q in range(n): qc.ry(th[i], q); i += 1
        for q in range(n - 1): qc.cz(q, q + 1)
    return qc, th


def c5_barren(samples=60):
    out = {}
    for cost in ["global", "local"]:
        out[cost] = {}
        for n in [2, 4, 6, 8, 10]:
            layers = n                     # depth grows with width
            Z = "Z" * n if cost == "global" else "I" * (n - 1) + "Z"
            op = SparsePauliOp(Z); grads = []
            for s in range(samples):
                rng = np.random.default_rng(s); qc, th = hea(n, layers, rng)
                def E(x):
                    c = QuantumCircuit(n); i = 0
                    for _ in range(layers):
                        for q in range(n): c.ry(x[i], q); i += 1
                        for q in range(n - 1): c.cz(q, q + 1)
                    return float(np.real(Statevector(c).expectation_value(op)))
                e0 = np.zeros_like(th); e0[0] = np.pi / 2
                grads.append((E(th + e0) - E(th - e0)) / 2)
            out[cost][n] = float(np.var(grads))
    p = plt(); fig, ax = p.subplots(figsize=(5.5, 4))
    for c in out: ax.semilogy(list(out[c]), list(out[c].values()), "o-", label=f"{c} cost")
    ax.set_xlabel("qubits (layers = qubits)"); ax.set_ylabel("Var[dE/dtheta_1]"); ax.legend()
    ax.set_title("Barren plateau: gradient variance"); fig.tight_layout(); fig.savefig(os.path.join(outdir(T), "C5_barren.png")); p.close(fig)
    ns = np.array(list(out["global"])); v = np.array(list(out["global"].values()))
    out["global_log2_slope_per_qubit"] = float(np.polyfit(ns, np.log2(v), 1)[0])
    return out


if __name__ == "__main__":
    r = {}
    cache = os.path.join(outdir(T), "_c1_cache.json")
    if os.path.exists(cache) and "--fresh" not in sys.argv:
        r["C1_qaoa"] = json.load(open(cache))
    else:
        r["C1_qaoa"] = c1_qaoa(); json.dump(r["C1_qaoa"], open(cache, "w"), default=float)
    print("C1 done", flush=True)
    r["C2_shots"], th = c2_shots(); print("C2 done", flush=True)
    r["C3_noise"] = c3_noise(th); print("C3 done", flush=True)
    r["C4_vqe"] = c4_vqe(); print("C4 done", flush=True)
    r["C5_barren"] = c5_barren(); print("C5 done", flush=True)
    r["note"] = "All results are from classical simulation (Qiskit Aer). Noise results use a synthetic noise model, not a device."
    save_json(T, "quantum_opt.json", r)
    print(json.dumps(r, indent=1, default=str))
