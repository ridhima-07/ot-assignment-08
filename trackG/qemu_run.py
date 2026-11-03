"""Boot xv6 under QEMU non-interactively, run shell commands, capture output.

Usage: python3 qemu_run.py <xv6_dir> [--smp N] [--timeout S] "cmd1" "cmd2" ...
Each command is typed at the xv6 shell prompt; output up to the next prompt is captured.
"""
import argparse, os, select, subprocess, sys, time

PROMPT = b"$ "


def boot(xv6_dir, smp=1, mem="128M"):
    cmd = ["qemu-system-riscv64", "-machine", "virt", "-bios", "none",
           "-kernel", "kernel/kernel", "-m", mem, "-smp", str(smp), "-nographic",
           "-global", "virtio-mmio.force-legacy=false",
           "-drive", "file=fs.img,if=none,format=raw,id=x0",
           "-device", "virtio-blk-device,drive=x0,bus=virtio-mmio-bus.0"]
    return subprocess.Popen(cmd, cwd=xv6_dir, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)


def read_until(p, token, timeout):
    buf = b""
    end = time.time() + timeout
    while time.time() < end:
        r, _, _ = select.select([p.stdout], [], [], 0.2)
        if r:
            chunk = os.read(p.stdout.fileno(), 4096)
            if not chunk:
                break
            buf += chunk
            if buf.endswith(token):
                return buf, True
        if p.poll() is not None:
            break
    return buf, False


def run(xv6_dir, commands, smp=1, timeout=600):
    p = boot(xv6_dir, smp)
    out, ok = read_until(p, PROMPT, 60)
    results = {"boot_ok": ok, "boot_log": out.decode(errors="replace"), "commands": []}
    if not ok:
        p.kill(); return results
    for c in commands:
        p.stdin.write(c.encode() + b"\n")
        t0 = time.time()
        o, ok = read_until(p, PROMPT, timeout)
        results["commands"].append({"cmd": c, "ok": ok, "secs": round(time.time() - t0, 2),
                                    "output": o.decode(errors="replace")})
        if not ok:
            break
    p.kill(); p.wait()
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("xv6_dir"); ap.add_argument("cmds", nargs="*")
    ap.add_argument("--smp", type=int, default=1); ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()
    res = run(a.xv6_dir, a.cmds, a.smp, a.timeout)
    print("BOOT_OK" if res["boot_ok"] else "BOOT_FAIL")
    for c in res["commands"]:
        print(f"### {c['cmd']}  ok={c['ok']} secs={c['secs']}\n{c['output']}")
    sys.exit(0 if res["boot_ok"] and all(c["ok"] for c in res["commands"]) else 1)
