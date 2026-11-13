"""
task3b_gemm.py -- Phase 3B: Progressive GEMM Optimization & 10-Point Tile-Size Sweep
Student: Ainat Yermurat (230103161)

C = A x B, double precision, N = 384, flat 1-D SharedMemory buffers
(3 x 384^2 x 8 B = 3.54 MB), rows partitioned across p = P_phys workers.
  V1 naive i-j-k      : dot += A[i*N+k] * B[k*N+j]  (B traversed with stride N)
  V2 reordered i-k-j  : a_ik hoisted, unit-stride j over rows of B and C
  V3 6-loop blocked   : (ii,kk,jj) tiles of BS, micro-kernel (i,k,j), a_ik hoisted
                        BS in {4,8,12,16,24,32,48,64,96,128}; verified vs V2 (< 1e-9)
Roofline ceilings are read from results/phase3a.json (run task3a_roofline.py first).
Emits VERIFY_HASH[PHASE3] using GFLOP/s(V1) and GFLOP/s(V3, BS*).
"""
import argparse
import random
import struct
from multiprocessing import shared_memory

from harness import (STUDENT_ID, emit_verify_tag, header, run_benchmark, ProcGroup, attach_shm,
                     save_json, load_json, common_args, resolve_pphys, get_cache_info, fmt_bytes)

N = 384
BS_LIST = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
I_NAIVE = 0.125


def _open(na, nb, nc):
    sa, sb, sc = attach_shm(na), attach_shm(nb), attach_shm(nc)
    return (sa, sb, sc), sa.buf.cast("d"), sb.buf.cast("d"), sc.buf.cast("d")


def _close(shms, *views):
    for v in views:
        v.release()
    for s in shms:
        s.close()


def gemm_v1(na, nb, nc, n, r0, r1, bs):
    shms, A, B, C = _open(na, nb, nc)
    for i in range(r0, r1):                       # rows partitioned across workers
        iN = i * n
        for j in range(n):                        # middle loop j
            dot = 0.0
            for k in range(n):                    # inner k, B with stride N
                dot += A[iN + k] * B[k * n + j]
            C[iN + j] = dot
    _close(shms, A, B, C)


def gemm_v2(na, nb, nc, n, r0, r1, bs):
    shms, A, B, C = _open(na, nb, nc)
    for i in range(r0, r1):
        iN = i * n
        for j in range(n):
            C[iN + j] = 0.0
        for k in range(n):                        # middle loop k
            a_ik = A[iN + k]                      # hoisted
            kN = k * n
            for j in range(n):                    # unit-stride rows of B and C
                C[iN + j] += a_ik * B[kN + j]
    _close(shms, A, B, C)


def gemm_v3(na, nb, nc, n, r0, r1, bs):
    shms, A, B, C = _open(na, nb, nc)
    for idx in range(r0 * n, r1 * n):
        C[idx] = 0.0
    for ii in range(r0, r1, bs):                          # loop 1: i tiles
        i_end = min(ii + bs, r1)
        for kk in range(0, n, bs):                        # loop 2: k tiles
            k_end = min(kk + bs, n)
            for jj in range(0, n, bs):                    # loop 3: j tiles
                j_end = min(jj + bs, n)
                for i in range(ii, i_end):                # loop 4: i  (micro-kernel i,k,j)
                    iN = i * n
                    for k in range(kk, k_end):            # loop 5: k
                        a_ik = A[iN + k]                  # hoisted outside j
                        kN = k * n
                        for j in range(jj, j_end):        # loop 6: j, unit stride
                            C[iN + j] += a_ik * B[kN + j]
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
    ap = common_args(argparse.ArgumentParser(), warmup=2, runs=7)
    args = ap.parse_args()
    header("PHASE 3B -- Progressive GEMM & 10-Point Tile Sweep")
    p = resolve_pphys(args)
    cache = get_cache_info()
    roof = load_json("phase3a.json")
    if roof is None:
        raise SystemExit("results/phase3a.json not found -- run task3a_roofline.py first.")
    BW, P, ridge = roof["BW_GBps"], roof["P_GFLOPs"], roof["ridge"]
    print(f"p = P_phys = {p}  N = {N}  footprint = {3 * N * N * 8 / 1e6:.2f} MB  "
          f"warmup_runs={args.warmup} timed_runs={args.runs}")
    print(f"Roofline (Phase 3A): BW = {BW:.2f} GB/s, P = {P:.4f} GFLOP/s, I* = {ridge:.6f} FLOP/B")
    print(f"L1d = {fmt_bytes(cache['l1d'])} per core, L2 = {fmt_bytes(cache['l2'])}, "
          f"L3 = {fmt_bytes(cache['l3'])}")

    nbytes = N * N * 8
    rnd = random.Random(STUDENT_ID)
    A_vals = [rnd.uniform(-1.0, 1.0) for _ in range(N * N)]
    B_vals = [rnd.uniform(-1.0, 1.0) for _ in range(N * N)]
    shA = shared_memory.SharedMemory(create=True, size=nbytes)
    shB = shared_memory.SharedMemory(create=True, size=nbytes)
    shC = shared_memory.SharedMemory(create=True, size=nbytes)
    flops = 2 * N ** 3
    rows = row_parts(N, p)

    def gflops(ms):
        return flops / (ms * 1e-3 * 1e9)

    try:
        shA.buf[:nbytes] = struct.pack(f"{N * N}d", *A_vals)
        shB.buf[:nbytes] = struct.pack(f"{N * N}d", *B_vals)

        def read_C():
            return struct.unpack(f"{N * N}d", bytes(shC.buf[:nbytes]))

        def bench(fn, bs=0):
            shC.buf[:nbytes] = bytes(nbytes)
            with ProcGroup(fn, [(shA.name, shB.name, shC.name, N, r0, r1, bs) for r0, r1 in rows]) as g:
                st = run_benchmark(g.run, args.warmup, args.runs)
            return st, read_C()

        def ref(i, j):
            return sum(A_vals[i * N + k] * B_vals[k * N + j] for k in range(N))

        print(f"\n{'variant':<20} {'median_ms':>10} {'IQR':>8} {'GFLOP/s':>9} {'W_tile':>10} "
              f"{'I FLOP/B':>9} {'%attain':>8} {'max|dC|':>10}")
        st1, C1 = bench(gemm_v1)
        g1 = gflops(st1.median_ms)
        st2, C2 = bench(gemm_v2)
        g2 = gflops(st2.median_ms)
        d12 = max(abs(x - y) for x, y in zip(C1, C2))
        dref = max(abs(C2[i * N + j] - ref(i, j)) for i, j in [(0, 0), (N // 5, (3 * N) // 4), (N - 1, N - 1), (N // 2, 5)])
        att_naive = min(P, I_NAIVE * BW)
        print(f"{'V1 naive i-j-k':<20} {st1.median_ms:>10.1f} {st1.iqr_ms:>8.1f} {g1:>9.5f} {'-':>10} "
              f"{I_NAIVE:>9.3f} {g1 / att_naive * 100:>7.1f}% {'(ref)':>10}")
        print(f"{'V2 reordered i-k-j':<20} {st2.median_ms:>10.1f} {st2.iqr_ms:>8.1f} {g2:>9.5f} {'-':>10} "
              f"{I_NAIVE:>9.3f} {g2 / att_naive * 100:>7.1f}% {d12:>10.2e}", flush=True)
        v3 = []
        for bs in BS_LIST:
            st3, C3 = bench(gemm_v3, bs)
            g3 = gflops(st3.median_ms)
            d32 = max(abs(x - y) for x, y in zip(C3, C2))
            wt = 3 * bs * bs * 8
            I = bs / 12
            att = min(P, I * BW)
            ok = d32 < 1e-9
            v3.append({"BS": bs, "median_ms": st3.median_ms, "iqr_ms": st3.iqr_ms, "gflops": g3,
                       "W_tile_B": wt, "I": I, "attainable": att, "pct_attain": g3 / att * 100,
                       "max_diff_vs_V2": d32, "ok": ok,
                       "fits_L1d": bool(cache["l1d"] and wt <= cache["l1d"]),
                       "fits_L2": bool(cache["l2"] and wt <= cache["l2"])})
            print(f"{'V3 BS=' + str(bs):<20} {st3.median_ms:>10.1f} {st3.iqr_ms:>8.1f} {g3:>9.5f} "
                  f"{fmt_bytes(wt):>10} {I:>9.3f} {g3 / att * 100:>7.1f}% {d32:>10.2e}"
                  f"{'' if ok else '  FAIL'}", flush=True)
            if not ok:
                raise SystemExit(f"Verification failed: BS={bs} max|C3-C2|={d32}")
    finally:
        for s in (shA, shB, shC):
            s.close(); s.unlink()

    best = max(v3, key=lambda r: r["gflops"])
    bs_star, g3s = best["BS"], best["gflops"]
    print(f"\nVerification: max|C2-C1| = {d12:.2e}; V2 vs independent dot products = {dref:.2e}; "
          f"all 10 tiles max|C3-C2| < 1e-9: {all(r['ok'] for r in v3)}")
    print(f"BS* = {bs_star}: {g3s:.5f} GFLOP/s, W_tile = {fmt_bytes(best['W_tile_B'])}, "
          f"I_tiled = BS*/12 = {bs_star / 12:.2f} FLOP/B")
    if cache["l1d"]:
        print(f"Tiles fitting L1d ({fmt_bytes(cache['l1d'])}): {[r['BS'] for r in v3 if r['fits_L1d']]}"
              f"   -> BS* fits L1d: {best['fits_L1d']}")
    if cache["l2"]:
        print(f"Tiles fitting L2  ({fmt_bytes(cache['l2'])}): {[r['BS'] for r in v3 if r['fits_L2']]}"
              f"   -> BS* fits L2: {best['fits_L2']}")
    print(f"Speedups: V2/V1 = {g2 / g1:.3f}x  V3*/V1 = {g3s / g1:.3f}x  V3*/V2 = {g3s / g2:.3f}x")

    print("\nRoofline mapping: attainable = min(P_sustained, I x BW_sustained)")
    for name, I, g in [("Variant 1", I_NAIVE, g1), ("Variant 2", I_NAIVE, g2),
                       (f"Variant 3 (BS*={bs_star})", bs_star / 12, g3s)]:
        att = min(P, I * BW)
        region = "compute(interpreter)-bound" if I >= ridge else "memory-bound"
        print(f"  {name:<22} I={I:7.3f}  attainable={att:.4f}  measured={g:.5f} GFLOP/s  "
              f"-> {g / att * 100:5.1f}% of attainable, {region}")

    tag = emit_verify_tag("PHASE3", STUDENT_ID, g1, g3s)
    print("\n" + tag)
    save_json("phase3b.json", {"P_phys": p, "N": N, "BW": BW, "P": P, "ridge": ridge,
                               "V1": {"median_ms": st1.median_ms, "iqr_ms": st1.iqr_ms, "gflops": g1},
                               "V2": {"median_ms": st2.median_ms, "iqr_ms": st2.iqr_ms, "gflops": g2,
                                      "max_diff_vs_V1": d12, "max_diff_vs_ref": dref},
                               "V3": v3, "BS_star": bs_star, "cache": cache, "verify": tag})


if __name__ == "__main__":
    main()
