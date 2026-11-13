"""
task4_scheduling.py -- Phase 4: Skewed Workloads, Load Imbalance & Dynamic Work Stealing
Student: Ainat Yermurat (230103161)

Triangular skew: N = 12,000 items, item i runs K(i) = 1 + floor((i/60)^2)
iterations of (math.sin, math.cos, math.sqrt); p = P_phys worker processes.
  Strategy 1: contiguous block static  [t*N/p, (t+1)*N/p)
  Strategy 2: cyclic round-robin static  i = t, t+p, t+2p, ...
  Strategy 3: shared-memory dynamic chunk queue, mp.Value('i', 0) counter,
              C in {1, 4, 16, 64, 256, 1024}
Each worker writes its active time t_worker[t] (ms) to a shared mp.Array('d', p).

Fairness: the same persistent workers execute every configuration, and the 8
configurations are run in INTERLEAVED rounds (round r runs all 8 once), so slow
thermal / turbo drift of the laptop affects all strategies equally instead of
penalising whichever strategy happens to run last.
"""
import argparse
import math
import multiprocessing as mp
import time

from harness import (STUDENT_ID, emit_verify_tag, header, ProcGroup, summarize,
                     save_json, common_args, resolve_pphys)

N_ITEMS = 12_000
CHUNKS = [1, 4, 16, 64, 256, 1024]
MODE_BLOCK, MODE_CYCLIC, MODE_DYNAMIC = 0, 1, 2


def K(i: int) -> int:
    return 1 + (i * i) // 3600            # exact integer form of 1 + floor((i/60)^2)


def work_item(i: int) -> float:
    sin, cos, sqrt = math.sin, math.cos, math.sqrt
    x = (i + 1) * 1e-3
    for _ in range(K(i)):
        x = sin(x) + cos(x)
        x = sqrt(x * x + 1.0)
    return x


def sched_worker(t, p, times, counter, mode, chunk):
    """One scheduling run for worker t; strategy selected by shared mode/chunk values."""
    m, C = mode.value, chunk.value
    acc = 0.0
    t0 = time.perf_counter()
    if m == MODE_BLOCK:
        for i in range(t * N_ITEMS // p, (t + 1) * N_ITEMS // p):
            acc += work_item(i)
    elif m == MODE_CYCLIC:
        for i in range(t, N_ITEMS, p):
            acc += work_item(i)
    else:
        lock = counter.get_lock()
        while True:
            with lock:                                 # synchronized shared index
                start = counter.value
                counter.value = start + C
            if start >= N_ITEMS:
                break
            for i in range(start, min(start + C, N_ITEMS)):
                acc += work_item(i)
    times[t] = (time.perf_counter() - t0) * 1e3        # ms


def main():
    ap = common_args(argparse.ArgumentParser(), warmup=1, runs=5)
    args = ap.parse_args()
    header("PHASE 4 -- Skewed Workloads, Load Imbalance & Dynamic Work Stealing")
    p = resolve_pphys(args)
    total = sum(K(i) for i in range(N_ITEMS))
    print(f"p = P_phys = {p}  N = {N_ITEMS:,}  K(0) = {K(0)}  K(N-1) = {K(N_ITEMS - 1):,}  "
          f"total iterations = {total:,}")
    print(f"warmup rounds = {args.warmup}, timed rounds = {args.runs} (all 8 configs interleaved per round)")
    blk = [sum(K(i) for i in range(t * N_ITEMS // p, (t + 1) * N_ITEMS // p)) for t in range(p)]
    cyc = [sum(K(i) for i in range(t, N_ITEMS, p)) for t in range(p)]
    print(f"Analytical lambda from K(i): block = {max(blk) / (sum(blk) / p):.3f}, "
          f"cyclic = {max(cyc) / (sum(cyc) / p):.4f}")

    configs = [("Strategy 1: block static", MODE_BLOCK, 0),
               ("Strategy 2: cyclic static", MODE_CYCLIC, 0)] + \
              [(f"Strategy 3: dynamic C={c}", MODE_DYNAMIC, c) for c in CHUNKS]

    times = mp.Array("d", p)
    counter = mp.Value("i", 0)
    mode = mp.Value("i", 0)
    chunk = mp.Value("i", 1)
    samples = {name: [] for name, _, _ in configs}

    with ProcGroup(sched_worker, [(t, p, times, counter, mode, chunk) for t in range(p)]) as g:
        def run_once(m, c):
            mode.value, chunk.value, counter.value = m, max(c, 1), 0
            for t in range(p):
                times[t] = 0.0
            t0 = time.perf_counter_ns()
            g.run()
            wall = (time.perf_counter_ns() - t0) / 1e6
            tw = list(times)
            return wall, max(tw) / (sum(tw) / p), tw

        for r in range(args.warmup + args.runs):
            tag_r = "warm-up" if r < args.warmup else f"timed {r - args.warmup + 1}/{args.runs}"
            for name, m, c in configs:
                res = run_once(m, c)
                if r >= args.warmup:
                    samples[name].append(res)
            print(f"  round {r + 1} ({tag_r}) done", flush=True)

    rows = []
    for name, m, c in configs:
        res = samples[name]
        st = summarize([x[0] for x in res])
        med = sorted(res, key=lambda x: x[0])[len(res) // 2]       # run with median T_wall
        rows.append({"name": name, "mode": m, "C": c or None, "median_ms": st.median_ms,
                     "iqr_ms": st.iqr_ms, "min_ms": st.min_ms, "max_ms": st.max_ms,
                     "lambda": med[1], "lambda_mean": sum(x[1] for x in res) / len(res),
                     "t_worker_ms": med[2], "sum_t_worker_ms": sum(med[2])})

    print(f"\n{'configuration':<27} {'T_wall ms':>10} {'IQR':>8} {'lambda':>7} {'sum t_w ms':>11}  t_worker[ms]")
    for r in rows:
        tw = " ".join(f"{x:.0f}" for x in r["t_worker_ms"])
        print(f"{r['name']:<27} {r['median_ms']:>10.1f} {r['iqr_ms']:>8.1f} {r['lambda']:>7.3f} "
              f"{r['sum_t_worker_ms']:>11.0f}  [{tw}]")

    dyn = [r for r in rows if r["mode"] == MODE_DYNAMIC]
    best = min(dyn, key=lambda r: r["median_ms"])
    T1 = rows[0]["median_ms"]
    print(f"\nC* = {best['C']}: T_wall = {best['median_ms']:.1f} ms, lambda = {best['lambda']:.3f}; "
          f"speedup over block static = {T1 / best['median_ms']:.3f}x")
    c1 = next(r for r in dyn if r["C"] == 1)
    cbig = next(r for r in dyn if r["C"] == 1024)
    last_start = (N_ITEMS - 1) // 1024 * 1024
    tail_share = sum(K(i) for i in range(last_start, N_ITEMS)) / total * 100
    first_full = sum(K(i) for i in range(last_start - 1024, last_start)) / total * 100
    print(f"C=1    : {N_ITEMS:,} lock acquisitions; lambda = {c1['lambda']:.3f}; T = {c1['median_ms']:.1f} ms")
    print(f"C=1024 : {math.ceil(N_ITEMS / 1024)} chunks; final chunk items {last_start}..{N_ITEMS - 1} "
          f"= {tail_share:.1f}% of all work (previous chunk {first_full:.1f}%); "
          f"lambda = {cbig['lambda']:.3f}; T = {cbig['median_ms']:.1f} ms")
    print(f"Mean work per item = {total / N_ITEMS:,.0f} iterations")

    tag = emit_verify_tag("PHASE4", STUDENT_ID, T1, best["median_ms"])
    print("\n" + tag)
    save_json("phase4.json", {"P_phys": p, "rows": rows, "C_star": best["C"], "verify": tag})


if __name__ == "__main__":
    main()
