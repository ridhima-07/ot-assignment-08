"""Track G: copy-on-write fork, before (eager copy, original xv6) vs after, same kernel (runtime switch).
Fresh boot per repetition. Times from the RISC-V time CSR (10 MHz under QEMU) via memstat()."""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from trackG.qemu_run import run
from common.util import save_json, outdir, plt, mean_std

T = "trackG"
XV6 = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/ext/xv6-cow"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
KV = re.compile(r"(\w+)=([\w.]+)")


def main():
    rows = []
    for rep in range(REPS):
        cmds = [f"cowbench {c} {t}" for t in (4, 1, 2, 3) for c in (0, 1)]
        res = run(XV6, cmds, smp=1, timeout=900); assert res["boot_ok"]
        for c in res["commands"]:
            for line in c["output"].splitlines():
                if line.startswith("COW "):
                    d = dict(KV.findall(line)); d["rep"] = rep; rows.append(d)
    def pick(test, cow, field, **kw):
        xs = [float(r[field]) for r in rows if r["test"] == str(test) and r["cow"] == str(cow) and all(r.get(k) == str(v) for k, v in kw.items())]
        return mean_std(xs)
    out = {"isolation_test": {c: [r["RESULT"] for r in rows if r["test"] == "4" and r["cow"] == str(c)] for c in (0, 1)},
           "fork_exit_2MB_heap": {name: {"avg_us": pick(1, c, "avg_us"), "pages_copied_per_20_forks": pick(1, c, "pages_copied")}
                                  for c, name in [(0, "eager copy (original)"), (1, "copy-on-write")]},
           "child_writes_fraction": {f"{p}%": {name: {"avg_us": pick(2, c, "avg_us", write_pct=p), "pages_copied_per_5_forks": pick(2, c, "pages_copied", write_pct=p)}
                                               for c, name in [(0, "eager"), (1, "COW")]} for p in (0, 25, 50, 100)},
           "memory_8_children_pages": {name: pick(3, c, "pages_used_by_children") for c, name in [(0, "eager"), (1, "COW")]},
           "reps": REPS}
    save_json(T, "cow_bench.json", {"summary": out, "raw": rows})
    p = plt(); fig, axes = p.subplots(1, 2, figsize=(11, 4)); pcts = [0, 25, 50, 100]
    for name in ["eager", "COW"]:
        axes[0].errorbar(pcts, [out["child_writes_fraction"][f"{q}%"][name]["avg_us"][0] / 1000 for q in pcts],
                         yerr=[out["child_writes_fraction"][f"{q}%"][name]["avg_us"][1] / 1000 for q in pcts], marker="o", capsize=3, label=name)
    axes[0].set_xlabel("% of the 2 MB heap the child writes"); axes[0].set_ylabel("fork + child + wait (ms)"); axes[0].legend()
    axes[0].set_title("COW wins when children write little")
    m = out["memory_8_children_pages"]; axes[1].bar(list(m), [v[0] for v in m.values()]); axes[1].set_ylabel("physical pages used by 8 live children")
    fig.tight_layout(); fig.savefig(os.path.join(outdir(T), "G_cow.png")); p.close(fig)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
