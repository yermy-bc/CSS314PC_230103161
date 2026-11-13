"""
task3_roofline_gemm.py -- Phase 3: Empirical Roofline & Cache-Blocked GEMM
Student: Ainat Yermurat (230103161)

Part 3A  (p = P_phys)
  * DRAM bandwidth: two 64 MB SharedMemory byte buffers; each worker copies its
    slice  mv_a[s:e] = mv_b[s:e].  BW = 2*64e6 / (T_median * 1e9)  [GB/s]
  * Compute ceiling: unrolled 16-variable scalar FMA loop, 6.0e7 FLOPs/worker.
    P = p*6e7 / (T_median*1e9) [GFLOP/s];  ridge point I* = P / BW  [FLOP/byte]

Part 3B  (N = 256, p = P_phys), double precision, flat 1-D shared buffers
  V1 naive i-j-k | V2 reordered i-k-j | V3 6-loop blocked (i,k,j micro-kernel)
"""
import argparse
import multiprocessing as mp
import random
import struct
from multiprocessing import shared_memory

from harness import (STUDENT_ID, emit_verify_tag, header, bench_procs, attach_shm,
                     save_json, common_args, resolve_pphys, get_cache_info)

BW_BYTES = 64 * 10 ** 6
FLOPS_PER_WORKER = 60_000_000
N = 256
BS_LIST = [8, 16, 32, 64, 96, 128]


# ======================= Part 3A workers =======================
def bw_worker(name_a, name_b, s, e, barrier=None):
    sa, sb = attach_shm(name_a), attach_shm(name_b)
    mv_a, mv_b = sa.buf, sb.buf
    barrier.wait()
    mv_a[s:e] = mv_b[s:e]
    barrier.wait()
    del mv_a, mv_b
    sa.close(); sb.close()


def fma_worker(iters, barrier=None):
    # 16 independent accumulators -> 16 FMAs (32 FLOPs) per iteration
    a = 1.0000001
    b = 1e-9
    x0 = x1 = x2 = x3 = x4 = x5 = x6 = x7 = 1.0
    x8 = x9 = x10 = x11 = x12 = x13 = x14 = x15 = 1.0
    barrier.wait()
    for _ in range(iters):
        x0 = x0 * a + b; x1 = x1 * a + b; x2 = x2 * a + b; x3 = x3 * a + b
        x4 = x4 * a + b; x5 = x5 * a + b; x6 = x6 * a + b; x7 = x7 * a + b
        x8 = x8 * a + b; x9 = x9 * a + b; x10 = x10 * a + b; x11 = x11 * a + b
        x12 = x12 * a + b; x13 = x13 * a + b; x14 = x14 * a + b; x15 = x15 * a + b
    barrier.wait()
    return x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15


# ======================= Part 3B workers =======================
def _views(na, nb, nc):
    sa, sb, sc = attach_shm(na), attach_shm(nb), attach_shm(nc)
    return (sa, sb, sc), sa.buf.cast("d"), sb.buf.cast("d"), sc.buf.cast("d")


def _close(shms, *views):
    for v in views:
        v.release()
    for s in shms:
        s.close()


def gemm_v1(na, nb, nc, n, r0, r1, bs, barrier=None):
    shms, A, B, C = _views(na, nb, nc)
    barrier.wait()
    for i in range(r0, r1):                 # row-partitioned i
        iN = i * n
        for j in range(n):                  # middle j
            s = 0.0
            for k in range(n):              # strided inner k over B[k*N + j]
                s += A[iN + k] * B[k * n + j]
            C[iN + j] = s
    barrier.wait()
    _close(shms, A, B, C)


def gemm_v2(na, nb, nc, n, r0, r1, bs, barrier=None):
    shms, A, B, C = _views(na, nb, nc)
    barrier.wait()
    for i in range(r0, r1):                 # row-partitioned i
        iN = i * n
        for j in range(n):
            C[iN + j] = 0.0
        for k in range(n):                  # middle k
            a_ik = A[iN + k]                # hoisted
            kN = k * n
            for j in range(n):              # unit-stride inner j
                C[iN + j] += a_ik * B[kN + j]
    barrier.wait()
    _close(shms, A, B, C)


def gemm_v3(na, nb, nc, n, r0, r1, bs, barrier=None):
    shms, A, B, C = _views(na, nb, nc)
    barrier.wait()
    for idx in range(r0 * n, r1 * n):
        C[idx] = 0.0
    for ii in range(r0, r1, bs):                         # tile i
        i_end = min(ii + bs, r1)
        for kk in range(0, n, bs):                       # tile k
            k_end = min(kk + bs, n)
            for jj in range(0, n, bs):                   # tile j
                j_end = min(jj + bs, n)
                for i in range(ii, i_end):               # micro-kernel (i,k,j)
                    iN = i * n
                    for k in range(kk, k_end):
                        a_ik = A[iN + k]                 # scalar hoisting
                        kN = k * n
                        for j in range(jj, j_end):
                            C[iN + j] += a_ik * B[kN + j]
    barrier.wait()
    _close(shms, A, B, C)


def row_parts(n, p):
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
    header("PHASE 3 -- Empirical Roofline & Cache-Blocked GEMM")
    p = resolve_pphys(args)
    cache = get_cache_info()
    print(f"p = P_phys = {p}  warmup={args.warmup} runs={args.runs}")
    W, R = args.warmup, args.runs

    # ---------------------------- 3A ----------------------------
    print("\n[3A] Roofline ceilings")
    sa = shared_memory.SharedMemory(create=True, size=BW_BYTES)
    sb = shared_memory.SharedMemory(create=True, size=BW_BYTES)
    try:
        sb.buf[:BW_BYTES] = bytes([7]) * BW_BYTES      # touch source pages
        sa.buf[:BW_BYTES] = bytes(BW_BYTES)            # touch destination pages
        parts = row_parts(BW_BYTES, p)
        bw_st, _ = bench_procs(lambda: (bw_worker, [(sa.name, sb.name, s, e) for s, e in parts]), W, R)
        BW = 2 * BW_BYTES / (bw_st.median_ms * 1e-3 * 1e9)
        assert sa.buf[BW_BYTES - 1] == 7
    finally:
        sa.close(); sa.unlink(); sb.close(); sb.unlink()
    print(f"DRAM copy 64 MB x {p} workers: median {bw_st.median_ms:.2f} ms (IQR {bw_st.iqr_ms:.2f})"
          f"  ->  BW_sustained = {BW:.2f} GB/s")

    iters = FLOPS_PER_WORKER // 32
    fma_st, _ = bench_procs(lambda: (fma_worker, [(iters,)] * p), W, R)
    P = p * FLOPS_PER_WORKER / (fma_st.median_ms * 1e-3 * 1e9)
    ridge = P / BW
    print(f"FMA loop {FLOPS_PER_WORKER:.1e} FLOP x {p} workers: median {fma_st.median_ms:.1f} ms"
          f"  ->  P_sustained = {P:.4f} GFLOP/s")
    print(f"Ridge point I* = P/BW = {ridge:.6f} FLOP/byte")

    # ---------------------------- 3B ----------------------------
    print(f"\n[3B] GEMM N={N}, double, flat 1-D buffers ({3 * N * N * 8 / 1e6:.2f} MB), p={p}")
    nbytes = N * N * 8
    rnd = random.Random(STUDENT_ID)
    A_vals = [rnd.uniform(-1, 1) for _ in range(N * N)]
    B_vals = [rnd.uniform(-1, 1) for _ in range(N * N)]
    shA = shared_memory.SharedMemory(create=True, size=nbytes)
    shB = shared_memory.SharedMemory(create=True, size=nbytes)
    shC = shared_memory.SharedMemory(create=True, size=nbytes)
    flops = 2 * N ** 3
    rows = row_parts(N, p)
    results = {}
    try:
        shA.buf[:nbytes] = struct.pack(f"{N*N}d", *A_vals)
        shB.buf[:nbytes] = struct.pack(f"{N*N}d", *B_vals)

        def read_C():
            return struct.unpack(f"{N*N}d", bytes(shC.buf[:nbytes]))

        def run(fn, bs=0):
            st, _ = bench_procs(lambda: (fn, [(shA.name, shB.name, shC.name, N, r0, r1, bs)
                                              for r0, r1 in rows]), W, R)
            return st

        # spot-check against an independent pure-Python reference (a few entries)
        def ref(i, j):
            return sum(A_vals[i * N + k] * B_vals[k * N + j] for k in range(N))

        print(f"{'variant':<22} {'median_ms':>10} {'IQR':>8} {'GFLOP/s':>9} {'W_tile KB':>10} {'I~BS/12':>8} {'max|dC|':>10}")
        st1 = run(gemm_v1); C1 = read_C()
        g1 = flops / (st1.median_ms * 1e-3 * 1e9)
        st2 = run(gemm_v2); C2 = read_C()
        g2 = flops / (st2.median_ms * 1e-3 * 1e9)
        d12 = max(abs(x - y) for x, y in zip(C1, C2))
        dref = max(abs(C2[i * N + j] - ref(i, j)) for i, j in [(0, 0), (17, 200), (255, 255), (128, 3)])
        print(f"{'V1 naive i-j-k':<22} {st1.median_ms:>10.1f} {st1.iqr_ms:>8.1f} {g1:>9.5f} {'-':>10} {'-':>8} {'ref':>10}")
        print(f"{'V2 reordered i-k-j':<22} {st2.median_ms:>10.1f} {st2.iqr_ms:>8.1f} {g2:>9.5f} {'-':>10} {'-':>8} {d12:>10.2e}")
        results["V1"] = {"median_ms": st1.median_ms, "iqr_ms": st1.iqr_ms, "gflops": g1}
        results["V2"] = {"median_ms": st2.median_ms, "iqr_ms": st2.iqr_ms, "gflops": g2,
                         "max_diff_vs_V1": d12, "max_diff_vs_ref": dref}
        v3 = []
        for bs in BS_LIST:
            st3 = run(gemm_v3, bs); C3 = read_C()
            g3 = flops / (st3.median_ms * 1e-3 * 1e9)
            d32 = max(abs(x - y) for x, y in zip(C3, C2))
            ok = d32 < 1e-9
            wt = 3 * bs * bs * 8
            v3.append({"BS": bs, "median_ms": st3.median_ms, "iqr_ms": st3.iqr_ms, "gflops": g3,
                       "W_tile_B": wt, "I_tiled": bs / 12, "max_diff_vs_V2": d32, "ok": ok})
            print(f"{'V3 blocked BS=' + str(bs):<22} {st3.median_ms:>10.1f} {st3.iqr_ms:>8.1f} {g3:>9.5f} "
                  f"{wt / 1024:>10.1f} {bs / 12:>8.2f} {d32:>10.2e}{'' if ok else '  FAIL'}", flush=True)
            if not ok:
                raise SystemExit(f"Verification failed for BS={bs}: max|C3-C2| = {d32}")
        results["V3"] = v3
    finally:
        for s in (shA, shB, shC):
            s.close(); s.unlink()

    best = max(v3, key=lambda r: r["gflops"])
    bs_star, g3s = best["BS"], best["gflops"]
    print(f"\nVerification: max|C2-C1| = {d12:.2e}, spot-check vs reference = {dref:.2e}, "
          f"all V3 tiles max|C3-C2| < 1e-9: {all(r['ok'] for r in v3)}")
    print(f"BS* = {bs_star}  (GFLOP/s = {g3s:.5f}),  W_tile(BS*) = {best['W_tile_B'] / 1024:.1f} KiB, "
          f"I_tiled(BS*) ~ {bs_star / 12:.2f} FLOP/B")
    l1, l2 = cache["l1d"], cache["l2"]
    if l1:
        print(f"L1d = {l1 / 1024:.0f} KiB -> tiles fitting L1d: "
              f"{[r['BS'] for r in v3 if r['W_tile_B'] <= l1]}")
    if l2:
        print(f"L2  = {l2 / 1024:.0f} KiB -> tiles fitting L2 : "
              f"{[r['BS'] for r in v3 if r['W_tile_B'] <= l2]}")
    print(f"Speedup V2/V1 = {g2 / g1:.3f}x   V3(BS*)/V1 = {g3s / g1:.3f}x   V3(BS*)/V2 = {g3s / g2:.3f}x")

    # Roofline placement: attainable = min(P, I*BW)
    print("\nRoofline mapping (attainable = min(P_sustained, I x BW)):")
    for name, I, g in [("V1 (I~0.125: 1 B-load of 8B/2 FLOP)", 2 / 16, g1),
                       ("V2 (I~0.25: streaming B row)", 2 / 8, g2),
                       (f"V3 BS*={bs_star} (I~BS/12)", bs_star / 12, g3s)]:
        att = min(P, I * BW)
        bound = "compute(interp)-bound" if I > ridge else "memory-bound"
        print(f"  {name:<38} I={I:7.3f}  attainable={att:8.4f}  measured={g:8.5f} GFLOP/s  "
              f"({g / P * 100:5.1f}% of P)  -> {bound}")

    tag = emit_verify_tag("PHASE3", STUDENT_ID, g1, g3s)
    print("\n" + tag)
    save_json("phase3.json", {"P_phys": p, "BW_GBps": BW, "bw_median_ms": bw_st.median_ms,
                              "P_GFLOPs": P, "fma_median_ms": fma_st.median_ms, "ridge": ridge,
                              "gemm": results, "BS_star": bs_star, "cache": cache, "verify": tag})


if __name__ == "__main__":
    main()
