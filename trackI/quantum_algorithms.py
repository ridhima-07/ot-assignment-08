"""Track I: quantum algorithms and QML vs classical. Everything is SIMULATED (Qiskit Aer / NumPy);
the "hardware" run uses the FakeManilaV2 noise model (calibration snapshot of a 5-qubit IBM device),
not a real device. All circuits below are written by hand (no library algorithm classes).

  I1 Deutsch-Jozsa      I2 Bernstein-Vazirani     I3 Grover (+ vs linear search, + noisy device model)
  I4 QFT (vs DFT matrix) I5 QPE (error vs counting qubits)   I6 Shor (factor 15, order finding by QPE)
  I7 Quantum annealing: exact adiabatic simulation on the Track B Max-Cut, spectral gap,
     and minor-embedding limits on D-Wave Chimera / Pegasus graphs
  I8 QML: variational classifier (data re-uploading) and quantum-kernel SVM vs logistic, RBF-SVM, MLP
"""
import sys, os, json, time, math
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Operator
from qiskit_aer import AerSimulator
from common.util import save_json, outdir, plt, mean_std, RESULTS

T = "trackI"
SIM = AerSimulator(seed_simulator=7)


def counts(qc, shots=2048, backend=SIM):
    return backend.run(transpile(qc, backend, optimization_level=1), shots=shots).result().get_counts()


# ---------------- I1 Deutsch-Jozsa ----------------
def dj_oracle(n, kind, rng):
    qc = QuantumCircuit(n + 1)
    if kind == "const1": qc.x(n)
    elif kind == "balanced":
        mask = rng.integers(0, 2, n); mask[rng.integers(n)] = 1          # f(x) = mask . x (mod 2), balanced
        flip = rng.integers(0, 2, n)
        for i in range(n):
            if flip[i]: qc.x(i)
        for i in range(n):
            if mask[i]: qc.cx(i, n)
        for i in range(n):
            if flip[i]: qc.x(i)
    return qc


def deutsch_jozsa(n=5, trials=4):
    rng = np.random.default_rng(0); res = []
    for kind in ["const0", "const1"] + ["balanced"] * trials:
        qc = QuantumCircuit(n + 1, n); qc.x(n); qc.h(range(n + 1))
        qc.compose(dj_oracle(n, kind, rng), inplace=True); qc.h(range(n)); qc.measure(range(n), range(n))
        c = counts(qc, 512); zero = c.get("0" * n, 0) / 512
        res.append({"oracle": kind, "P(all zeros)": zero, "verdict": "constant" if zero > 0.5 else "balanced"})
    ok = all((r["verdict"] == "constant") == r["oracle"].startswith("const") for r in res)
    return {"n": n, "runs": res, "all_correct": ok, "quantum_queries": 1, "classical_worst_case_queries": 2 ** (n - 1) + 1}


# ---------------- I2 Bernstein-Vazirani ----------------
def bernstein_vazirani(secret="101101"):
    n = len(secret); qc = QuantumCircuit(n + 1, n); qc.x(n); qc.h(range(n + 1))
    for i, b in enumerate(reversed(secret)):
        if b == "1": qc.cx(i, n)
    qc.h(range(n)); qc.measure(range(n), range(n))
    c = counts(qc, 256); got = max(c, key=c.get)
    return {"secret": secret, "measured": got, "correct": got == secret, "quantum_queries": 1, "classical_queries": n}


# ---------------- I3 Grover ----------------
def grover_circuit(n, marked, iters, measure=True):
    qc = QuantumCircuit(n, n if measure else 0); qc.h(range(n))
    def phase_flip(bits):                           # flip the phase of |bits>
        for i, b in enumerate(reversed(bits)):
            if b == "0": qc.x(i)
        if n == 1: qc.z(0)
        else: qc.h(n - 1); qc.mcx(list(range(n - 1)), n - 1); qc.h(n - 1)
        for i, b in enumerate(reversed(bits)):
            if b == "0": qc.x(i)
    for _ in range(iters):
        phase_flip(marked)                          # oracle
        qc.h(range(n)); phase_flip("0" * n); qc.h(range(n))   # diffusion (= -(2|s><s| - I), global phase)
    if measure: qc.measure(range(n), range(n))
    return qc


def grover():
    out = {}; rng = np.random.default_rng(1)
    for n in range(2, 11):
        N = 2 ** n; marked = format(int(rng.integers(N)), f"0{n}b"); k = int(math.floor(math.pi / 4 * math.sqrt(N)))
        qc = grover_circuit(n, marked, k, measure=False)
        t0 = time.perf_counter(); p = float(abs(Statevector(qc).data[int(marked, 2)]) ** 2); tsim = time.perf_counter() - t0
        tq = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1)
        ops = tq.count_ops()
        # classical linear search: expected oracle queries (N+1)/2, measured on random targets
        t0 = time.perf_counter(); qs = []
        for _ in range(200):
            target = int(rng.integers(N)); order = rng.permutation(N); qs.append(int(np.argmax(order == target)) + 1)
        tcl = (time.perf_counter() - t0) / 200
        out[n] = {"N": N, "grover_iterations(queries)": k, "success_prob": p, "classical_mean_queries": float(np.mean(qs)),
                  "query_ratio_classical/grover": float(np.mean(qs)) / k, "cx_gates": ops.get("cx", 0), "depth": tq.depth(),
                  "statevector_sim_secs": tsim, "classical_search_secs": tcl}
    # noisy device model: n = 3 on FakeManila (5 qubits, real calibration data)
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    dev = AerSimulator.from_backend(FakeManilaV2()); qc = grover_circuit(3, "101", 2)
    ideal = counts(qc, 4096).get("101", 0) / 4096; noisy = counts(qc, 4096, dev).get("101", 0) / 4096
    tq = transpile(qc, FakeManilaV2(), optimization_level=3, seed_transpiler=1)
    out["noisy_device_model_n3"] = {"ideal_P(marked)": ideal, "FakeManila_P(marked)": noisy, "random_guess": 1 / 8,
                                    "transpiled_cx": tq.count_ops().get("cx", 0) + tq.count_ops().get("ecr", 0), "transpiled_depth": tq.depth()}
    ns = list(range(2, 11))
    p = plt(); fig, axes = p.subplots(1, 2, figsize=(11, 4))
    axes[0].semilogy([2 ** n for n in ns], [out[n]["classical_mean_queries"] for n in ns], "o-", label="classical linear search (measured mean)")
    axes[0].semilogy([2 ** n for n in ns], [out[n]["grover_iterations(queries)"] for n in ns], "s-", label="Grover oracle queries")
    axes[0].set_xscale("log", base=2); axes[0].set_xlabel("N = 2^n"); axes[0].set_ylabel("oracle queries"); axes[0].legend()
    axes[1].semilogy(ns, [out[n]["statevector_sim_secs"] for n in ns], "o-", label="simulating Grover (statevector)")
    axes[1].semilogy(ns, [out[n]["classical_search_secs"] for n in ns], "s-", label="classical linear search")
    axes[1].set_xlabel("qubits n"); axes[1].set_ylabel("wall-clock seconds"); axes[1].legend(); axes[1].set_title("query advantage is not a wall-clock advantage in simulation", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(outdir(T), "I3_grover.png")); p.close(fig)
    return out


# ---------------- I4 QFT ----------------
def qft(n, inverse=False):
    qc = QuantumCircuit(n)
    for j in reversed(range(n)):
        qc.h(j)
        for k in reversed(range(j)): qc.cp(math.pi / 2 ** (j - k), k, j)
    for i in range(n // 2): qc.swap(i, n - 1 - i)
    return qc.inverse() if inverse else qc


def qft_check(n=4):
    U = Operator(qft(n)).data; N = 2 ** n
    F = np.array([[np.exp(2j * np.pi * j * k / N) for k in range(N)] for j in range(N)]) / np.sqrt(N)
    err = float(np.max(np.abs(U - F)))
    # period finding demo: state with period 4 -> peaks at multiples of N/4
    psi = np.zeros(N); psi[::4] = 1; psi /= np.linalg.norm(psi)
    out = Statevector(psi).evolve(qft(n)).probabilities()
    peaks = [int(i) for i in np.flatnonzero(out > 1e-9)]
    return {"n": n, "max_abs_diff_vs_DFT_matrix": err, "period4_input_peaks": peaks, "expected_peaks": list(range(0, N, N // 4)),
            "gate_count": sum(qft(n).count_ops().values()), "classical_FFT_ops_~NlogN": N * n}


# ---------------- I5 QPE ----------------
def qpe(phi, t):
    qc = QuantumCircuit(t + 1, t); qc.x(t); qc.h(range(t))
    for k in range(t): qc.cp(2 * math.pi * phi * 2 ** k, k, t)       # controlled-U^(2^k), U = P(2 pi phi), |1> eigenstate
    qc.compose(qft(t, inverse=True), range(t), inplace=True); qc.measure(range(t), range(t))
    c = counts(qc, 2048); best = max(c, key=c.get)
    return int(best, 2) / 2 ** t, c[best] / 2048


def qpe_study():
    out = {}
    for phi in [0.3125, 1 / 3]:
        out[f"phi={phi:.6f}"] = {}
        for t in range(3, 9):
            est, p = qpe(phi, t)
            out[f"phi={phi:.6f}"][t] = {"estimate": est, "abs_error": abs(est - phi), "P(best outcome)": p, "resolution_2^-t": 2 ** -t}
    return out


# ---------------- I6 Shor (N = 15) ----------------
def c_amod15(a, power):
    """Controlled multiplication by a^power mod 15 on 4 work qubits (standard construction for a coprime to 15)."""
    U = QuantumCircuit(4)
    for _ in range(power):
        if a in (2, 13): U.swap(2, 3); U.swap(1, 2); U.swap(0, 1)
        if a in (7, 8): U.swap(0, 1); U.swap(1, 2); U.swap(2, 3)
        if a in (4, 11): U.swap(1, 3); U.swap(0, 2)
        if a in (7, 11, 13):
            for q in range(4): U.x(q)
    return U.to_gate(label=f"{a}^{power} mod 15").control()


def shor15(a, t=8):
    qc = QuantumCircuit(t + 4, t); qc.h(range(t)); qc.x(t)            # work register starts in |1>
    for k in range(t): qc.append(c_amod15(a, 2 ** k), [k] + list(range(t, t + 4)))
    qc.compose(qft(t, inverse=True), range(t), inplace=True); qc.measure(range(t), range(t))
    c = counts(qc, 1024); found = {}
    for bits, n in sorted(c.items(), key=lambda kv: -kv[1])[:6]:
        phase = int(bits, 2) / 2 ** t; r = Fraction(phase).limit_denominator(15).denominator
        f = None
        if r % 2 == 0 and pow(a, r // 2, 15) != 14:
            g = {math.gcd(pow(a, r // 2) - 1, 15), math.gcd(pow(a, r // 2) + 1, 15)} - {1, 15}
            f = sorted(g) if g else None
        found[bits] = {"phase": phase, "order_guess": r, "factors": f, "freq": n / 1024}
    true_r = next(r for r in range(1, 16) if pow(a, r, 15) == 1)
    return {"a": a, "true_order": true_r, "outcomes": found, "factored": any(v["factors"] == [3, 5] for v in found.values())}


# ---------------- I7 quantum annealing ----------------
def annealing():
    inst = json.load(open(os.path.join(RESULTS, "trackB", "maxcut_instance.json")))
    n = inst["n"]; J = np.array(inst["J"]); h = np.array(inst["h"]); c0 = inst["c"]
    Z = np.array([[1, 0], [0, -1]]); X = np.array([[0, 1], [1, 0]]); I = np.eye(2)
    def op(single, i):
        m = np.array([[1.0]])
        for k in range(n): m = np.kron(m, single if k == i else I)
        return m
    Zs = [np.diag(op(Z, i)).real for i in range(n)]
    Hp = np.zeros(2 ** n)
    for i in range(n):
        Hp += h[i] * Zs[i]
        for j in range(n): Hp += J[i, j] * Zs[i] * Zs[j]
    Hp += c0                                                            # equals x^T Q x = -cut
    Hd = -sum(op(X, i) for i in range(n))
    ground = np.flatnonzero(np.isclose(Hp, Hp.min()))
    gaps = []
    for s in np.linspace(0, 1, 41):
        ev = np.linalg.eigvalsh((1 - s) * Hd + s * np.diag(Hp)); gaps.append(ev[len(ground)] - ev[0])   # gap above the degenerate ground space
    res = {"qubits": n, "min_cut_energy": float(Hp.min()), "optimum_cut": inst["optimum"], "ground_states": len(ground),
           "min_gap_along_path": float(min(gaps[:-1])), "by_anneal_time": {}}
    psi0 = np.ones(2 ** n) / np.sqrt(2 ** n)
    for Ttot in [0.5, 2, 5, 10, 20, 40]:
        steps = int(max(40, Ttot * 20)); dt = Ttot / steps; psi = psi0.astype(complex)
        for k in range(steps):
            s = (k + 0.5) / steps; psi = expm(-1j * dt * ((1 - s) * Hd + s * np.diag(Hp))) @ psi
        p = float(np.sum(np.abs(psi[ground]) ** 2))
        res["by_anneal_time"][Ttot] = {"P(optimal cut)": p, "expected_cut": float(-(np.abs(psi) ** 2) @ Hp)}
    # embedding limits on annealing hardware graphs
    import minorminer, dwave_networkx as dnx, networkx as nx
    emb = {}
    for hw_name, H in [("Chimera C(4) 128 qubits", dnx.chimera_graph(4)), ("Pegasus P(6) 680 qubits", dnx.pegasus_graph(6))]:
        rows = {}
        G = nx.Graph(); G.add_edges_from(inst["edges"])
        for name, src in [("Track B Max-Cut (8 nodes)", G)] + [(f"K{k}", nx.complete_graph(k)) for k in [6, 10, 14, 18, 24, 32]]:
            e = minorminer.find_embedding(list(src.edges), list(H.edges), random_seed=0, timeout=20)
            if e: rows[name] = {"physical_qubits": sum(len(v) for v in e.values()), "max_chain": max(len(v) for v in e.values()),
                                "qubits_per_logical": sum(len(v) for v in e.values()) / src.number_of_nodes()}
            else: rows[name] = {"embedded": False}
        emb[hw_name] = rows
    res["embedding"] = emb
    return res


# ---------------- I8 QML vs classical ----------------
def _ry(t): c, s = np.cos(t / 2), np.sin(t / 2); return np.array([[c, -s], [s, c]], dtype=complex)
def _rz(t): return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]])


def vqc_state(x, th, L):
    """2-qubit data re-uploading circuit, NumPy statevector (validated against Qiskit below).
    Per layer: RY(x_q) RY(th) RZ(th) on each qubit, then CZ."""
    psi = np.zeros(4, complex); psi[0] = 1; CZ = np.diag([1, 1, 1, -1])
    for l in range(L):
        g = []
        for q in range(2):
            g.append(_rz(th[l, q, 1]) @ _ry(th[l, q, 0]) @ _ry(x[q]))
        psi = np.kron(g[1], g[0]) @ psi; psi = CZ @ psi                 # qubit 0 is the least significant (Qiskit order)
    return psi


def vqc_expect(x, th, L):
    psi = vqc_state(x, th, L); return float(np.real(np.sum(np.abs(psi) ** 2 * np.array([1, -1, 1, -1]))))   # <Z_0>


def vqc_qiskit(x, th, L):
    qc = QuantumCircuit(2)
    for l in range(L):
        for q in range(2): qc.ry(x[q], q); qc.ry(th[l, q, 0], q); qc.rz(th[l, q, 1], q)
        qc.cz(0, 1)
    from qiskit.quantum_info import SparsePauliOp
    return float(np.real(Statevector(qc).expectation_value(SparsePauliOp("IZ"))))


def train_vqc(X, y, L=3, epochs=60, lr=0.1, seed=0):
    rng = np.random.default_rng(seed); th = rng.normal(0, 0.5, (L, 2, 2)); evals = 0
    m = np.zeros_like(th); v = np.zeros_like(th); b1, b2 = 0.9, 0.999
    t = 2 * y - 1
    for ep in range(1, epochs + 1):
        f = np.array([vqc_expect(x, th, L) for x in X]); evals += len(X)
        r = f - t; g = np.zeros_like(th)
        for idx in np.ndindex(th.shape):                                 # parameter-shift rule, exact
            e = np.zeros_like(th); e[idx] = np.pi / 2
            fp = np.array([vqc_expect(x, th + e, L) for x in X]); fm = np.array([vqc_expect(x, th - e, L) for x in X]); evals += 2 * len(X)
            g[idx] = np.mean(r * (fp - fm) / 2) * 2
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        th -= lr * (m / (1 - b1 ** ep)) / (np.sqrt(v / (1 - b2 ** ep)) + 1e-8)
    return th, evals


def zz_state(x):
    """ZZ feature map on 2 qubits (2 repetitions), NumPy statevector."""
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2); psi = np.zeros(4, complex); psi[0] = 1
    for _ in range(2):
        psi = np.kron(H, H) @ psi
        ph = np.array([1, np.exp(2j * x[0]), np.exp(2j * x[1]), np.exp(2j * (x[0] + x[1]))])     # P(2x_i) per qubit
        zz = (np.pi - x[0]) * (np.pi - x[1]); zzp = np.array([1, np.exp(2j * zz), np.exp(2j * zz), 1])  # CX P CX
        psi = zzp * ph * psi
    return psi


def qml_vs_classical(seeds=5):
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    # simulator validation
    rng = np.random.default_rng(9); th = rng.normal(size=(3, 2, 2)); xs = rng.uniform(0, np.pi, (5, 2))
    vqc_err = max(abs(vqc_expect(x, th, 3) - vqc_qiskit(x, th, 3)) for x in xs)
    rows = {k: {"acc": [], "train_s": [], "evals": []} for k in ["VQC (data re-uploading, 12 params)", "Quantum kernel SVM (ZZ map)",
                                                                  "Logistic regression", "RBF-SVM", "MLP (16 hidden)"]}
    for s in range(seeds):
        X, y = make_moons(120, noise=0.15, random_state=s); Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=s)
        sc = MinMaxScaler((0, np.pi)).fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        t0 = time.perf_counter(); th, ev = train_vqc(Xtr, ytr, seed=s); tt = time.perf_counter() - t0
        # training target is t = 2y - 1, i.e. class 1 <-> <Z> = +1, so predict class 1 when <Z> > 0.
        # (Bug fixed: the first version used "< 0", which inverted every label and reported 6.7% accuracy.)
        pred = np.array([vqc_expect(x, th, 3) for x in Xte]) > 0
        K = rows["VQC (data re-uploading, 12 params)"]; K["acc"].append(np.mean(pred == yte)); K["train_s"].append(tt); K["evals"].append(ev + len(Xte))
        t0 = time.perf_counter(); Str = np.array([zz_state(x) for x in Xtr]); Ste = np.array([zz_state(x) for x in Xte])
        Ktr = np.abs(Str.conj() @ Str.T) ** 2; Kte = np.abs(Ste.conj() @ Str.T) ** 2
        svm = SVC(kernel="precomputed").fit(Ktr, ytr); tt = time.perf_counter() - t0
        K = rows["Quantum kernel SVM (ZZ map)"]; K["acc"].append(svm.score(Kte, yte)); K["train_s"].append(tt)
        K["evals"].append(len(Xtr) * (len(Xtr) - 1) // 2 + len(Xte) * len(Xtr)); K.setdefault("support_vectors", []).append(int(svm.n_support_.sum()))
        for name, mdl in [("Logistic regression", LogisticRegression()), ("RBF-SVM", SVC()), ("MLP (16 hidden)", MLPClassifier((16,), max_iter=3000, random_state=s))]:
            t0 = time.perf_counter(); mdl.fit(Xtr, ytr); tt = time.perf_counter() - t0
            rows[name]["acc"].append(mdl.score(Xte, yte)); rows[name]["train_s"].append(tt); rows[name]["evals"].append(0)
    params = {"VQC (data re-uploading, 12 params)": 12, "Quantum kernel SVM (ZZ map)": "n_support (dual)", "Logistic regression": 3,
              "RBF-SVM": "n_support (dual)", "MLP (16 hidden)": 2 * 16 + 16 + 16 + 1}
    out = {name: {"test_acc_mean": mean_std(d["acc"])[0], "test_acc_std": mean_std(d["acc"])[1], "train_secs_mean": mean_std(d["train_s"])[0],
                  "circuit_evaluations_mean": float(np.mean(d["evals"])), "trainable_params": params[name]} for name, d in rows.items()}
    out["simulator_validation_max_abs_diff_vs_qiskit"] = float(vqc_err)
    out["note"] = "Quantum models run on an exact statevector simulator; circuit evaluations are what a real device would have to execute (each needing many shots)."
    return out


if __name__ == "__main__":
    r = {}
    r["I1_deutsch_jozsa"] = deutsch_jozsa(); print("I1", r["I1_deutsch_jozsa"]["all_correct"], flush=True)
    r["I2_bernstein_vazirani"] = bernstein_vazirani(); print("I2", r["I2_bernstein_vazirani"], flush=True)
    r["I3_grover"] = grover(); print("I3", r["I3_grover"]["noisy_device_model_n3"], flush=True)
    r["I4_qft"] = qft_check(); print("I4", r["I4_qft"], flush=True)
    r["I5_qpe"] = qpe_study(); print("I5 done", flush=True)
    r["I6_shor"] = {a: shor15(a) for a in [7, 2, 4, 11, 13]}; print("I6", {a: v["factored"] for a, v in r["I6_shor"].items()}, flush=True)
    r["I7_annealing"] = annealing(); print("I7", r["I7_annealing"]["by_anneal_time"], flush=True)
    r["I8_qml_vs_classical"] = qml_vs_classical(); print("I8", json.dumps(r["I8_qml_vs_classical"], indent=1), flush=True)
    r["status"] = "simulation + FakeManilaV2 noise model; no real quantum hardware was used"
    save_json(T, "quantum_algorithms.json", r)
