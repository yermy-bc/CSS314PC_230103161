"""
task4_scheduling.py -- Phase 4: Skewed Workloads & Dynamic Chunk Queue
Student: Ainat Yermurat (230103161)

Triangular skewed workload: N = 8000 items, item i runs
K(i) = 1 + floor((i/40)^2) heavy iterations (sin, cos, sqrt), p = P_phys workers.
Strategies: (1) contiguous block static, (2) cyclic round-robin static,
(3) shared-memory dynamic queue (mp.Value('i', 0) counter) with C in {1, 8, 64, 512}.
Each worker's active runtime goes into a shared mp.Array('d', p).
"""
import argparse
import math
import multiprocessing as mp
import time

from harness import (STUDENT_ID, emit_verify_tag, header, run_procs_timed, summarize,
                     save_json, common_args, resolve_pphys)

N_ITEMS = 8000
CHUNKS = [1, 8, 64, 512]


def K(i: int) -> int:
    return 1 + (i * i) // 1600          # == 1 + floor((i/40)^2), exact integer form


def work_item(i: int) -> float:
    sin, cos, sqrt = math.sin, math.cos, math.sqrt
    x = (i + 1) * 1e-3
    for _ in range(K(i)):
        x = sin(x) + cos(x)
        x = sqrt(x * x + 1.0)
    return x


def static_block_worker(t, p, times, barrier=None):
    q, r = divmod(N_ITEMS, p)
    start = t * q + min(t, r)
    end = start + q + (1 if t < r else 0)
    barrier.wait()
    t0 = time.perf_counter()
    acc = 0.0
    for i in range(start, end):
        acc += work_item(i)
    times[t] = time.perf_counter() - t0
    barrier.wait()


def static_cyclic_worker(t, p, times, barrier=None):
    barrier.wait()
    t0 = time.perf_counter()
    acc = 0.0
    for i in range(t, N_ITEMS, p):
        acc += work_item(i)
    times[t] = time.perf_counter() - t0
    barrier.wait()


def dynamic_worker(t, p, times, counter, C, barrier=None):
    barrier.wait()
    t0 = time.perf_counter()
    acc = 0.0
    lock = counter.get_lock()
    while True:
        with lock:                      # synchronized index counter
            start = counter.value
            counter.value = start + C
        if start >= N_ITEMS:
            break
        for i in range(start, min(start + C, N_ITEMS)):
            acc += work_item(i)
    times[t] = time.perf_counter() - t0
    barrier.wait()


def lam(times):
    ts = list(times)
    return max(ts) / (sum(ts) / len(ts))


def run_strategy(name, p, warmup, runs, C=None):
    times = mp.Array("d", p)
    counter = mp.Value("i", 0)

    def once():
        for t in range(p):
            times[t] = 0.0
        if name == "block":
            tgt, a = static_block_worker, [(t, p, times) for t in range(p)]
        elif name == "cyclic":
            tgt, a = static_cyclic_worker, [(t, p, times) for t in range(p)]
        else:
            counter.value = 0
            tgt, a = dynamic_worker, [(t, p, times, counter, C) for t in range(p)]
        ms = run_procs_timed(tgt, a)
        return ms, lam(times), list(times)

    for _ in range(warmup):
        once()
    res = [once() for _ in range(runs)]
    st = summarize([r[0] for r in res])
    med_run = sorted(res, key=lambda r: r[0])[len(res) // 2]   # run whose T_wall is the median
    return {"strategy": name, "C": C, "median_ms": st.median_ms, "iqr_ms": st.iqr_ms,
            "min_ms": st.min_ms, "lambda": med_run[1],
            "lambda_all": [r[1] for r in res], "t_worker_s": med_run[2]}


def main():
    ap = common_args(argparse.ArgumentParser(), warmup=1, runs=5)
    args = ap.parse_args()
    header("PHASE 4 -- Skewed Workloads & Dynamic Chunk Queue")
    p = resolve_pphys(args)
    total = sum(K(i) for i in range(N_ITEMS))
    print(f"p = P_phys = {p}  N = {N_ITEMS}  total iterations = {total:,}  "
          f"K(0)={K(0)} K(N-1)={K(N_ITEMS - 1)}  warmup={args.warmup} runs={args.runs}")
    # share of work in the last contiguous block -> expected static-block imbalance
    q = N_ITEMS // p
    blk = [sum(K(i) for i in range(t * q, (t + 1) * q if t < p - 1 else N_ITEMS)) for t in range(p)]
    print(f"Ideal block-static lambda (from K) = {max(blk) / (sum(blk) / p):.3f}")

    rows = []
    rows.append(run_strategy("block", p, args.warmup, args.runs))
    rows.append(run_strategy("cyclic", p, args.warmup, args.runs))
    for C in CHUNKS:
        rows.append(run_strategy("dynamic", p, args.warmup, args.runs, C))

    print(f"\n{'strategy':<18} {'T_wall ms':>10} {'IQR':>8} {'lambda':>7}  t_worker[s]")
    for r in rows:
        label = r["strategy"] + (f" C={r['C']}" if r["C"] else "")
        tw = " ".join(f"{x:.2f}" for x in r["t_worker_s"])
        print(f"{label:<18} {r['median_ms']:>10.1f} {r['iqr_ms']:>8.1f} {r['lambda']:>7.3f}  [{tw}]")

    dyn = [r for r in rows if r["strategy"] == "dynamic"]
    best = min(dyn, key=lambda r: r["median_ms"])
    T_block = rows[0]["median_ms"]
    print(f"\nOptimal chunk C* = {best['C']}  (T_wall = {best['median_ms']:.1f} ms, lambda = {best['lambda']:.3f})")
    print(f"Speedup of dynamic C* over block static: {T_block / best['median_ms']:.3f}x")
    c1 = next(r for r in dyn if r["C"] == 1)
    c512 = next(r for r in dyn if r["C"] == 512)
    print(f"C=1   : {N_ITEMS} lock acquisitions, lambda={c1['lambda']:.3f}, T={c1['median_ms']:.1f} ms")
    print(f"C=512 : {math.ceil(N_ITEMS / 512)} chunks only, last chunk = items 7680..7999 "
          f"({sum(K(i) for i in range(7680, 8000)) / total * 100:.1f}% of all work), "
          f"lambda={c512['lambda']:.3f}, T={c512['median_ms']:.1f} ms")

    tag = emit_verify_tag("PHASE4", STUDENT_ID, T_block, best["median_ms"])
    print("\n" + tag)
    save_json("phase4.json", {"P_phys": p, "rows": rows, "C_star": best["C"], "verify": tag})


if __name__ == "__main__":
    main()
