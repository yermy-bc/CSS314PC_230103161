"""
task1_falsesharing.py -- Phase 1: Forensic Diagnosis of False Sharing & Cache-Line Contention
Student: Ainat Yermurat (230103161)

SharedMemory(create=True, size=4096) -> 512 x uint64 slots. Worker t updates slot
t*stride with M/p deterministic RMW steps (M = 2.4e7), using the fs_worker kernel
from the handout verbatim. Workers are persistent processes (harness.ProcGroup),
so run_benchmark(..., warmup_runs=2, timed_runs=7) measures only the kernel.
"""
import argparse
import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory

from harness import (STUDENT_ID, emit_verify_tag, header, run_benchmark, ProcGroup, attach_shm,
                     pow2_sweep, save_json, common_args, resolve_pphys, get_cache_info)

SHM_BYTES = 4096
SLOTS = 512
M_TOTAL = 24_000_000            # 2.4 x 10^7 RMW steps in total
STRIDES = [1, 2, 4, 8, 16, 32]  # 8, 16, 32, 64, 128, 256 bytes


def fs_worker(shm_name: str, slot_idx: int, iters: int) -> None:
    shm = attach_shm(shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    acc = arr[slot_idx]
    mask = 0xFFFFFFFFFFFFFFFF
    mult = 6364136223846793005
    for i in range(iters):
        acc = (acc * mult + (i | 1)) & mask
        arr[slot_idx] = acc  # Direct hardware store to shared L1 cache line
    del arr  # Release buffer export before closing SharedMemory
    shm.close()


def measure(shm, p, stride, warmup, runs):
    iters = M_TOTAL // p
    args = [(shm.name, (t * stride) % SLOTS, iters) for t in range(p)]
    with ProcGroup(fs_worker, args) as g:
        st = run_benchmark(g.run, warmup_runs=warmup, timed_runs=runs)
    mops = M_TOTAL / (st.median_ms * 1e-3) / 1e6
    return {"p": p, "stride": stride, "bytes": stride * 8, "min_ms": st.min_ms,
            "median_ms": st.median_ms, "iqr_ms": st.iqr_ms, "max_ms": st.max_ms, "mops": mops}


def karp_flatt(S, p):
    return (1.0 / S - 1.0 / p) / (1.0 - 1.0 / p)


def main():
    ap = common_args(argparse.ArgumentParser(), warmup=2, runs=7)
    args = ap.parse_args()
    header("PHASE 1 -- False Sharing & Cache-Line Contention (Karp-Flatt)")
    pphys = resolve_pphys(args)
    plog = mp.cpu_count()
    line = get_cache_info()["line"] or 64
    print(f"P_phys={pphys}  P_log={plog}  cache line={line} B  M={M_TOTAL:,}  "
          f"warmup_runs={args.warmup} timed_runs={args.runs}")

    shm = shared_memory.SharedMemory(create=True, size=SHM_BYTES)
    try:
        shm.buf[:SHM_BYTES] = bytes(SHM_BYTES)

        # ---------------- Table 1A ----------------
        print(f"\nTable 1A -- Stride sensitivity at p = P_phys = {pphys}")
        print(f"{'stride':>6} {'bytes':>6} {'median_ms':>11} {'IQR_ms':>9} {'min_ms':>10} {'max_ms':>10} {'Mops/s':>9}")
        t1a = []
        for s in STRIDES:
            r = measure(shm, pphys, s, args.warmup, args.runs)
            t1a.append(r)
            print(f"{s:>6} {s*8:>6} {r['median_ms']:>11.2f} {r['iqr_ms']:>9.2f} {r['min_ms']:>10.2f} "
                  f"{r['max_ms']:>10.2f} {r['mops']:>9.2f}", flush=True)
        drops = [((a["median_ms"] - b["median_ms"]) / a["median_ms"], b["bytes"])
                 for a, b in zip(t1a, t1a[1:])]
        d, at = max(drops)
        print(f"Sharpest drop: {d*100:.1f}% when slot separation reaches {at} B "
              f"(stride {at // 8}); hardware line = {line} B")
        t_s1 = t1a[0]["median_ms"]
        t_s16 = t1a[STRIDES.index(16)]["median_ms"]
        print(f"False-sharing penalty T(stride=1)/T(stride=16) at p={pphys}: {t_s1 / t_s16:.3f}x")

        # ---------------- Table 1B ----------------
        ps = pow2_sweep(plog)
        t1b = {}
        for s in (1, 16):
            print(f"\nTable 1B -- Karp-Flatt series, stride = {s} ({s*8} B)")
            print(f"{'p':>3} {'median_ms':>11} {'IQR_ms':>9} {'Mops/s':>9} {'S_p':>7} {'E_p':>7} {'e(p)':>8}")
            rows, T1 = [], None
            for p in ps:
                r = dict(next(x for x in t1a if x["stride"] == s)) if p == pphys \
                    else measure(shm, p, s, args.warmup, args.runs)
                if p == 1:
                    T1 = r["median_ms"]
                S = T1 / r["median_ms"]
                E = S / p
                e = karp_flatt(S, p) if p >= 2 else None
                r.update({"S": S, "E": E, "e": e})
                rows.append(r)
                es = f"{e:>8.4f}" if e is not None else f"{'-':>8}"
                print(f"{p:>3} {r['median_ms']:>11.2f} {r['iqr_ms']:>9.2f} {r['mops']:>9.2f} "
                      f"{S:>7.3f} {E:>7.3f} {es}", flush=True)
            t1b[s] = rows

        print("\nKarp-Flatt trajectory: stride=1 vs stride=16")
        print(f"{'p':>3} {'e(p) s=1':>10} {'e(p) s=16':>10} {'ratio':>7}")
        for a, b in zip(t1b[1], t1b[16]):
            if a["e"] is not None:
                ratio = a["e"] / b["e"] if b["e"] else float("nan")
                print(f"{a['p']:>3} {a['e']:>10.4f} {b['e']:>10.4f} {ratio:>7.2f}")

        tag = emit_verify_tag("PHASE1", STUDENT_ID, t_s1, t_s16)
        print("\n" + tag)
        save_json("phase1.json", {"P_phys": pphys, "P_log": plog, "table1A": t1a,
                                  "table1B": {str(k): v for k, v in t1b.items()}, "verify": tag})
    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()
