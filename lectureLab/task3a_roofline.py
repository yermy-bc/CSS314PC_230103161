"""
task3a_roofline.py -- Phase 3A: Characterizing the Laptop's Hardware Roofline
Student: Ainat Yermurat (230103161)

Across p = P_phys persistent worker processes:
 1. BW_sustained: two 96 MB SharedMemory byte buffers (192 MB > L3); each worker
    copies its slice  mv_a[start:end] = mv_b[start:end]  (native memcpy).
    BW = 2 * 96e6 / (T_median * 1e9)  GB/s
 2. P_sustained: 16 unrolled scalar FMAs (x_k = x_k*a + b), 8.0e7 FLOP per worker.
    P = p * 8e7 / (T_median * 1e9)  GFLOP/s
 3. Ridge point I* = P / BW  FLOP/byte
"""
import argparse
from multiprocessing import shared_memory

from harness import (STUDENT_ID, header, run_benchmark, ProcGroup, attach_shm, save_json,
                     common_args, resolve_pphys, get_cache_info, fmt_bytes)

BUF_BYTES = 96 * 10 ** 6
FLOPS_PER_WORKER = 80_000_000
FLOPS_PER_ITER = 32                       # 16 FMAs x 2 FLOP


def bw_worker(name_a, name_b, start, end):
    sa, sb = attach_shm(name_a), attach_shm(name_b)
    mv_a, mv_b = sa.buf, sb.buf
    mv_a[start:end] = mv_b[start:end]
    del mv_a, mv_b
    sa.close()
    sb.close()


def fma_worker(iters):
    a = 0.9999999
    b = 1e-9
    x0 = x1 = x2 = x3 = x4 = x5 = x6 = x7 = 1.0
    x8 = x9 = x10 = x11 = x12 = x13 = x14 = x15 = 1.0
    for _ in range(iters):
        x0 = x0 * a + b; x1 = x1 * a + b; x2 = x2 * a + b; x3 = x3 * a + b
        x4 = x4 * a + b; x5 = x5 * a + b; x6 = x6 * a + b; x7 = x7 * a + b
        x8 = x8 * a + b; x9 = x9 * a + b; x10 = x10 * a + b; x11 = x11 * a + b
        x12 = x12 * a + b; x13 = x13 * a + b; x14 = x14 * a + b; x15 = x15 * a + b
    return x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15


def parts(n, p):
    q, r = divmod(n, p)
    out, s = [], 0
    for w in range(p):
        e = s + q + (1 if w < r else 0)
        out.append((s, e))
        s = e
    return out


def main():
    ap = common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    header("PHASE 3A -- Empirical Machine Roofline")
    p = resolve_pphys(args)
    cache = get_cache_info()
    print(f"p = P_phys = {p}  warmup_runs={args.warmup} timed_runs={args.runs}  "
          f"L3 = {fmt_bytes(cache['l3'])}")

    # ---------------- 1. sustained DRAM bandwidth ----------------
    sa = shared_memory.SharedMemory(create=True, size=BUF_BYTES)
    sb = shared_memory.SharedMemory(create=True, size=BUF_BYTES)
    try:
        sb.buf[:BUF_BYTES] = bytes([0x5A]) * BUF_BYTES       # first-touch source
        sa.buf[:BUF_BYTES] = bytes(BUF_BYTES)                 # first-touch destination
        with ProcGroup(bw_worker, [(sa.name, sb.name, s, e) for s, e in parts(BUF_BYTES, p)]) as g:
            bw = run_benchmark(g.run, args.warmup, args.runs)
        ok = sa.buf[0] == 0x5A and sa.buf[BUF_BYTES - 1] == 0x5A
    finally:
        sa.close(); sa.unlink(); sb.close(); sb.unlink()
    BW = 2 * BUF_BYTES / (bw.median_ms * 1e-3 * 1e9)
    print(f"\n[1] DRAM: copy 96 MB (192 MB footprint) over {p} workers")
    print(f"    median {bw.median_ms:.2f} ms  IQR {bw.iqr_ms:.2f}  min {bw.min_ms:.2f}  max {bw.max_ms:.2f}"
          f"  copy verified: {ok}")
    print(f"    BW_sustained = 2 x 96e6 / (T x 1e9) = {BW:.2f} GB/s")

    # ---------------- 2. sustained compute ceiling ----------------
    iters = FLOPS_PER_WORKER // FLOPS_PER_ITER
    with ProcGroup(fma_worker, [(iters,)] * p) as g:
        fm = run_benchmark(g.run, args.warmup, args.runs)
    P = p * FLOPS_PER_WORKER / (fm.median_ms * 1e-3 * 1e9)
    print(f"\n[2] Compute: 16-way unrolled scalar FMA, {FLOPS_PER_WORKER:.1e} FLOP/worker x {p}")
    print(f"    median {fm.median_ms:.1f} ms  IQR {fm.iqr_ms:.1f}  min {fm.min_ms:.1f}  max {fm.max_ms:.1f}")
    print(f"    P_sustained = {P:.4f} GFLOP/s  ({P / p:.4f} GFLOP/s per core)")

    # ---------------- 3. ridge point ----------------
    ridge = P / BW
    print(f"\n[3] Ridge point I* = P/BW = {P:.4f}/{BW:.2f} = {ridge:.6f} FLOP/byte")
    print("    (Phase 3 VERIFY_HASH is emitted by task3b_gemm.py, which uses these ceilings)")

    save_json("phase3a.json", {"P_phys": p, "BW_GBps": BW, "bw_stats": vars(bw),
                               "P_GFLOPs": P, "fma_stats": vars(fm), "ridge": ridge})


if __name__ == "__main__":
    main()
