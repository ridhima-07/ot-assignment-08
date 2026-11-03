"""Track H: index selection on PostgreSQL 16 + TPC-H (SF 0.1, official dbgen).

PROBLEM. Given a weighted workload Q and candidate secondary indexes I with sizes s_i,
choose x_i in {0,1} to minimise sum_q w_q * cost(q | {i : x_i = 1}) subject to sum_i s_i x_i <= B.

COST ORACLE. PostgreSQL's own planner cost with HypoPG hypothetical indexes ("what-if"),
so candidates are evaluated without building them.

METHODS
  default   primary keys only (what the DBA gets without tuning)
  greedy    repeatedly add the index with the largest (what-if workload saving) / size that fits
  ILP       atomic-benefit formulation (HiGHS): y_{q,t,i} = query q uses index i on table t,
            y <= x_i, at most one index per (query, table), size budget; benefits b_{q,i} from what-if
  all       every candidate (no storage limit) as a reference
Then each configuration is BUILT for real, ANALYZEd and the workload is executed:
median latency of 15 warm runs, storage, and an md5 of every query's result set (must match default).
"""
import sys, os, json, time, hashlib, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import psycopg2
from scipy.optimize import milp, LinearConstraint, Bounds
from common.util import save_json, outdir, plt

T = "trackH"

# ---------------- workload (weight = executions per period) ----------------
WORKLOAD = {
    # OLTP-style lookups of an order-management application
    "cust_orders":  (50, "SELECT o_orderkey, o_totalprice, o_orderdate FROM orders WHERE o_custkey = 1234 ORDER BY o_orderdate"),
    "order_lines":  (50, "SELECT l_linenumber, l_partkey, l_quantity FROM lineitem WHERE l_orderkey = 54321"),
    "ship_week":    (20, "SELECT count(*), sum(l_extendedprice) FROM lineitem WHERE l_shipdate BETWEEN DATE '1995-03-01' AND DATE '1995-03-07'"),
    "supp_parts":   (20, "SELECT ps_partkey, ps_availqty FROM partsupp WHERE ps_suppkey = 77"),
    "part_lines":   (20, "SELECT l_orderkey, l_quantity FROM lineitem WHERE l_partkey = 4242"),
    "cust_segment": (10, "SELECT count(*) FROM customer WHERE c_nationkey = 7 AND c_mktsegment = 'BUILDING'"),
    "part_type":    (10, "SELECT p_partkey, p_name FROM part WHERE p_type = 'ECONOMY ANODIZED STEEL' AND p_size = 15"),
    "recent_orders": (5, "SELECT count(*) FROM orders WHERE o_orderdate >= DATE '1998-07-01' AND o_orderstatus = 'O'"),
    # TPC-H analytic queries (standard validation parameters)
    "tpch_q3": (1, """SELECT l_orderkey, sum(l_extendedprice*(1-l_discount)) AS revenue, o_orderdate, o_shippriority
        FROM customer, orders, lineitem WHERE c_mktsegment = 'BUILDING' AND c_custkey = o_custkey AND l_orderkey = o_orderkey
        AND o_orderdate < DATE '1995-03-15' AND l_shipdate > DATE '1995-03-15'
        GROUP BY l_orderkey, o_orderdate, o_shippriority ORDER BY revenue DESC, o_orderdate LIMIT 10"""),
    "tpch_q5": (1, """SELECT n_name, sum(l_extendedprice*(1-l_discount)) AS revenue FROM customer, orders, lineitem, supplier, nation, region
        WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey AND l_suppkey = s_suppkey AND c_nationkey = s_nationkey
        AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey AND r_name = 'ASIA'
        AND o_orderdate >= DATE '1994-01-01' AND o_orderdate < DATE '1995-01-01' GROUP BY n_name ORDER BY revenue DESC"""),
    "tpch_q6": (1, """SELECT sum(l_extendedprice*l_discount) AS revenue FROM lineitem WHERE l_shipdate >= DATE '1994-01-01'
        AND l_shipdate < DATE '1995-01-01' AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24"""),
    "tpch_q10": (1, """SELECT c_custkey, c_name, sum(l_extendedprice*(1-l_discount)) AS revenue FROM customer, orders, lineitem, nation
        WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey AND o_orderdate >= DATE '1993-10-01' AND o_orderdate < DATE '1994-01-01'
        AND l_returnflag = 'R' AND c_nationkey = n_nationkey GROUP BY c_custkey, c_name ORDER BY revenue DESC LIMIT 20"""),
    "tpch_q12": (1, """SELECT l_shipmode, count(*) FROM orders, lineitem WHERE o_orderkey = l_orderkey AND l_shipmode IN ('MAIL','SHIP')
        AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate AND l_receiptdate >= DATE '1994-01-01'
        AND l_receiptdate < DATE '1995-01-01' GROUP BY l_shipmode ORDER BY l_shipmode"""),
    "tpch_q14": (1, """SELECT 100.00*sum(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice*(1-l_discount) ELSE 0 END)
        / sum(l_extendedprice*(1-l_discount)) FROM lineitem, part WHERE l_partkey = p_partkey
        AND l_shipdate >= DATE '1995-09-01' AND l_shipdate < DATE '1995-10-01'"""),
}

# candidates derived from WHERE / JOIN / ORDER BY columns of the workload
CANDIDATES = [
    ("orders", "o_custkey"), ("orders", "o_custkey, o_orderdate"), ("orders", "o_orderdate"), ("orders", "o_orderstatus, o_orderdate"),
    ("lineitem", "l_shipdate"), ("lineitem", "l_partkey"), ("lineitem", "l_suppkey"), ("lineitem", "l_receiptdate"),
    ("lineitem", "l_shipdate, l_discount, l_quantity"), ("lineitem", "l_shipmode, l_receiptdate"),
    ("partsupp", "ps_suppkey"), ("customer", "c_nationkey, c_mktsegment"), ("customer", "c_mktsegment"),
    ("part", "p_type, p_size"), ("part", "p_size"), ("supplier", "s_nationkey"),
]


def conn():
    c = psycopg2.connect(dbname="tpch"); c.autocommit = True; cur = c.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS hypopg"); cur.execute("SET max_parallel_workers_per_gather = 0"); cur.execute("SET jit = off")
    return c, cur


def plan_cost(cur, sql):
    cur.execute("EXPLAIN (FORMAT JSON) " + sql); return float(cur.fetchone()[0][0]["Plan"]["Total Cost"])


def whatif(cur, config):
    """Weighted workload cost and per-query costs with the hypothetical indexes in `config`."""
    cur.execute("SELECT hypopg_reset()")
    for i in config:
        t, cols = CANDIDATES[i]; cur.execute(f"SELECT * FROM hypopg_create_index('CREATE INDEX ON {t} ({cols})')")
    per = {q: plan_cost(cur, sql) for q, (w, sql) in WORKLOAD.items()}
    cur.execute("SELECT hypopg_reset()")
    return sum(WORKLOAD[q][0] * c for q, c in per.items()), per


def hypo_sizes(cur):
    sizes = []
    for t, cols in CANDIDATES:
        cur.execute("SELECT hypopg_reset()")
        cur.execute(f"SELECT indexrelid FROM hypopg_create_index('CREATE INDEX ON {t} ({cols})')"); oid = cur.fetchone()[0]
        cur.execute("SELECT hypopg_relation_size(%s)", (oid,)); sizes.append(int(cur.fetchone()[0]))
    cur.execute("SELECT hypopg_reset()"); return sizes


def greedy(cur, sizes, budget):
    cfg, used = [], 0; base, _ = whatif(cur, cfg); evals = 1
    while True:
        best, bi = 0.0, None
        for i in range(len(CANDIDATES)):
            if i in cfg or used + sizes[i] > budget: continue
            c, _ = whatif(cur, cfg + [i]); evals += 1
            ratio = (base - c) / sizes[i]
            if ratio > best: best, bi, bc = ratio, i, c
        if bi is None: break
        cfg.append(bi); used += sizes[bi]; base = bc
    return cfg, evals


def ilp(cur, sizes, budget, single_benefit):
    """single_benefit[q][i] = cost(q | {}) - cost(q | {i}) (what-if)."""
    qs = list(WORKLOAD); n = len(CANDIDATES)
    pairs = [(q, i) for q in qs for i in range(n) if single_benefit[q][i] > 1e-6]
    nv = n + len(pairs); c = np.zeros(nv)
    for k, (q, i) in enumerate(pairs): c[n + k] = -WORKLOAD[q][0] * single_benefit[q][i]
    rows, lo, hi = [], [], []
    for k, (q, i) in enumerate(pairs):                       # y_{q,i} <= x_i
        r = np.zeros(nv); r[n + k] = 1; r[i] = -1; rows.append(r); lo.append(-np.inf); hi.append(0)
    for q in qs:                                             # one index per (query, table)
        for t in {CANDIDATES[i][0] for i in range(n)}:
            ks = [k for k, (qq, i) in enumerate(pairs) if qq == q and CANDIDATES[i][0] == t]
            if len(ks) > 1:
                r = np.zeros(nv); r[[n + k for k in ks]] = 1; rows.append(r); lo.append(0); hi.append(1)
    r = np.zeros(nv); r[:n] = sizes; rows.append(r); lo.append(0); hi.append(budget)    # storage budget
    t0 = time.perf_counter()
    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi), integrality=np.ones(nv), bounds=Bounds(0, 1))
    return [i for i in range(n) if res.x[i] > 0.5], time.perf_counter() - t0, -res.fun


def execute(cur, config, reps=15):
    """Build the configuration for real and run the workload. Returns latencies, storage, result hashes."""
    cur.execute("SELECT indexname FROM pg_indexes WHERE schemaname='public' AND indexname LIKE 'a8_%'")
    for (n,) in cur.fetchall(): cur.execute(f"DROP INDEX {n}")
    t0 = time.perf_counter()
    for i in config:
        t, cols = CANDIDATES[i]; cur.execute(f"CREATE INDEX a8_{i} ON {t} ({cols})")
    build = time.perf_counter() - t0
    cur.execute("ANALYZE")
    cur.execute("SELECT coalesce(sum(pg_relation_size(indexrelid)),0) FROM pg_stat_user_indexes WHERE indexrelname LIKE 'a8_%'")
    storage = int(cur.fetchone()[0])
    lat, hashes, plans = {}, {}, {}
    for q, (w, sql) in WORKLOAD.items():
        cur.execute(sql); rows = cur.fetchall()                                                           # warm-up + result
        # SQL defines row order only under ORDER BY, so compare the multiset of rows for every query, and the exact
        # sequence for queries with ORDER BY. (First version hashed the raw sequence and flagged supp_parts, which has
        # no ORDER BY, as "different" once an index scan changed the row order.)
        key = sorted(rows, key=repr) if "ORDER BY" not in sql.upper() else rows
        hashes[q] = hashlib.md5(repr(key).encode()).hexdigest()
        ts = []
        for _ in range(reps):
            t1 = time.perf_counter(); cur.execute(sql); cur.fetchall(); ts.append(time.perf_counter() - t1)
        lat[q] = statistics.median(ts) * 1000
        cur.execute("EXPLAIN (ANALYZE, COSTS OFF, TIMING OFF, SUMMARY OFF) " + sql)
        plans[q] = "\n".join(r[0] for r in cur.fetchall())
    weighted = sum(WORKLOAD[q][0] * lat[q] for q in WORKLOAD)
    return {"config": [f"{CANDIDATES[i][0]}({CANDIDATES[i][1]})" for i in config], "index_bytes": storage, "build_secs": build,
            "weighted_ms": weighted, "latency_ms": lat, "hashes": hashes, "plans": plans}


def main():
    c, cur = conn(); out = {}
    sizes = hypo_sizes(cur); total = sum(sizes)
    base_cost, base_per = whatif(cur, [])
    single = {q: {} for q in WORKLOAD}
    for i in range(len(CANDIDATES)):
        _, per = whatif(cur, [i])
        for q in WORKLOAD: single[q][i] = base_per[q] - per[q]
    out["candidates"] = [{"index": f"{t}({cl})", "est_bytes": s} for (t, cl), s in zip(CANDIDATES, sizes)]
    out["whatif_default_cost"] = base_cost
    configs = {"default (PK only)": []}
    sel = {}
    for frac in [0.10, 0.25, 0.50]:
        B = frac * total
        t0 = time.perf_counter(); g, ev = greedy(cur, sizes, B); tg = time.perf_counter() - t0
        il, ti, ilp_obj = ilp(cur, sizes, B, single)
        cg, _ = whatif(cur, g); ci, _ = whatif(cur, il)
        sel[f"{int(frac * 100)}%"] = {"budget_bytes": B,
            "greedy": {"config": g, "whatif_cost": cg, "whatif_saving_%": 100 * (1 - cg / base_cost), "secs": tg, "whatif_calls": ev, "bytes": sum(sizes[i] for i in g)},
            "ilp": {"config": il, "whatif_cost": ci, "whatif_saving_%": 100 * (1 - ci / base_cost), "secs": ti, "ilp_predicted_saving": ilp_obj, "bytes": sum(sizes[i] for i in il)}}
        configs[f"greedy @{int(frac * 100)}%"] = g; configs[f"ILP @{int(frac * 100)}%"] = il
        print(frac, "greedy", g, round(cg), "ilp", il, round(ci), flush=True)
    configs["all candidates"] = list(range(len(CANDIDATES)))
    out["selection"] = sel
    out["executed"] = {}
    for name, cfg in configs.items():
        r = execute(cur, cfg); out["executed"][name] = r
        print(name, round(r["weighted_ms"], 1), r["index_bytes"] // 1024, "KB", flush=True)
    ref = out["executed"]["default (PK only)"]["hashes"]
    for name, r in out["executed"].items():
        r["results_identical_to_default"] = r["hashes"] == ref
    execute(cur, [])   # leave the database clean
    # keep two plan examples, drop the rest from the JSON
    ex = {}
    for name in ["default (PK only)", "ILP @25%"]:
        ex[name] = {q: out["executed"][name]["plans"][q] for q in ["cust_orders", "tpch_q14"]}
    for r in out["executed"].values(): r.pop("plans")
    out["plan_examples"] = ex
    save_json(T, "index_selection.json", out)
    p = plt(); fig, ax = p.subplots(figsize=(8, 4.2)); names = list(out["executed"])
    ax.bar(range(len(names)), [out["executed"][n]["weighted_ms"] for n in names])
    for k, n in enumerate(names): ax.text(k, out["executed"][n]["weighted_ms"], f"{out['executed'][n]['index_bytes'] / 2**20:.1f} MB", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("weighted workload time (ms, measured)"); ax.set_title("Index selection on TPC-H SF0.1 (label = index storage)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir(T), "H_index_selection.png")); p.close(fig)


if __name__ == "__main__":
    main()
