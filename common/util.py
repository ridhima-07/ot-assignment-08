"""Shared helpers: seeding, timing, result saving and plotting style."""
import json, os, random, time
from contextlib import contextmanager

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def outdir(track):
    d = os.path.join(RESULTS, track)
    os.makedirs(d, exist_ok=True)
    return d


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_json(track, name, obj):
    path = os.path.join(outdir(track), name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)
    return path


@contextmanager
def timer():
    t = {"s": None}
    t0 = time.perf_counter()
    yield t
    t["s"] = time.perf_counter() - t0


def mean_std(xs):
    xs = np.asarray(xs, dtype=float)
    return float(xs.mean()), float(xs.std(ddof=1) if len(xs) > 1 else 0.0)


def plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as p
    p.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                       "axes.spines.top": False, "axes.spines.right": False})
    return p


def md_table(rows, headers):
    """Render a list of row-lists as a GitHub markdown table."""
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)
