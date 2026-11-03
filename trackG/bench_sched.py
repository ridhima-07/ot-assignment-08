"""Track G: before/after scheduler benchmark under QEMU (1 CPU so policies are comparable).

For each repetition we boot a fresh xv6 and run 3 workloads under RR (original xv6) and MLFQ.
Metrics (timer ticks): turnaround = etime - ctime, response = ftime - ctime, makespan, per job kind.
"""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from trackG.qemu_run import run
from common.util import save_json, outdir, plt, mean_std

T = "trackG"
XV6 = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/ext/xv6-riscv"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
PAT = re.compile(r"STAT w=(\d) pol=(\d) kind=(\w+) pid=(\d+) ctime=(-?\d+) ftime=(-?\d+) etime=(-?\d+) rtime=(\d+)")
WL = {1: "W1 mixed batch (6 CPU jobs arrive together)", 2: "W2 2 CPU hogs + 3 interactive", 3: "W3 short jobs arriving behind 2 long jobs"}


def main():
    rows = []
    for rep in range(REPS):
        cmds = [f"schedbench {pol} {w}" for w in (1, 2, 3) for pol in (0, 1)]
        res = run(XV6, cmds, smp=1, timeout=900)
        assert res["boot_ok"], "xv6 failed to boot"
        for c in res["commands"]:
            for m in PAT.finditer(c["output"]):
                w, pol, kind, pid, ct, ft, et, rt = m.groups()
                rows.append(dict(rep=rep, w=int(w), pol=int(pol), kind=kind, turnaround=int(et) - int(ct),
                                 response=int(ft) - int(ct), etime=int(et), rtime=int(rt)))
        print("rep", rep, "rows", len(rows), flush=True)
    out = {}
    for w in (1, 2, 3):
        out[WL[w]] = {}
        for pol, name in [(0, "RR (original)"), (1, "MLFQ")]:
            sub = [r for r in rows if r["w"] == w and r["pol"] == pol]
            per_rep = lambda f: [f([r for r in sub if r["rep"] == k]) for k in range(REPS)]
            d = {"avg_turnaround": mean_std(per_rep(lambda rs: np.mean([r["turnaround"] for r in rs]))),
                 "avg_response": mean_std(per_rep(lambda rs: np.mean([r["response"] for r in rs]))),
                 "makespan": mean_std(per_rep(lambda rs: max(r["etime"] for r in rs)))}
            for kind in sorted({r["kind"] for r in sub}):
                d[f"{kind}_turnaround"] = mean_std(per_rep(lambda rs: np.mean([r["turnaround"] for r in rs if r["kind"] == kind])))
                d[f"{kind}_response"] = mean_std(per_rep(lambda rs: np.mean([r["response"] for r in rs if r["kind"] == kind])))
            out[WL[w]][name] = d
    save_json(T, "sched_bench.json", {"reps": REPS, "smp": 1, "units": "timer ticks (~0.1 s)", "summary": out, "raw": rows})
    p = plt(); fig, axes = p.subplots(1, 3, figsize=(14, 4))
    for ax, w in zip(axes, (1, 2, 3)):
        s = out[WL[w]]; keys = [k for k in s["RR (original)"] if k.endswith("turnaround") or k.endswith("response")]
        x = np.arange(len(keys)); wd = 0.38
        for j, pol in enumerate(["RR (original)", "MLFQ"]):
            ax.bar(x + (j - 0.5) * wd, [s[pol][k][0] for k in keys], wd, yerr=[s[pol][k][1] for k in keys], label=pol, capsize=2)
        ax.set_xticks(x); ax.set_xticklabels(keys, rotation=35, ha="right", fontsize=7); ax.set_title(WL[w], fontsize=9); ax.set_ylabel("ticks")
    axes[0].legend(); fig.tight_layout(); fig.savefig(os.path.join(outdir(T), "G_sched.png")); p.close(fig)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
