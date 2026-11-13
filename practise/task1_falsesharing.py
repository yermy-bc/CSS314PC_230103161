"""
task1_falsesharing.py -- Phase 1: False Sharing Stride Sweep & Karp-Flatt Diagnostics
Student: Ainat Yermurat (230103161)

A 4096-byte SharedMemory block holds 512 uint64 slots. Worker w owns slot
w*stride and performs M/p read-modify-write steps on it (Listing 2 kernel).
Timing is barrier-to-barrier inside the parent, so process spawn cost is excluded.
"""
import argparse
import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory

from harness import (STUDENT_ID, emit_verify_tag, header, bench_procs, attach_shm,
                     pow2_sweep, save_json, common_args, resolve_pphys, get_cache_info)

SHM_BYTES = 4096
SLOTS = SHM_BYTES // 8          # 512 x uint64
M_TOTAL = 16_000_000            # 1.6 x 10^7 total RMW steps
STRIDES = [1, 2, 4, 8, 16, 32]  # 8 .. 256 bytes


def fs_worker(shm_name: str, slot_idx: int, iters: int, barrier=None) -> None:
    shm = attach_shm(shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    barrier.wait()                                   # ---- timed region starts
    acc = arr[slot_idx]
    for i in range(iters):
        acc = (acc * 6364136223846793005 + (i | 1)) & 0xFFFFFFFFFFFFFFFF
        arr[slot_idx] = acc  # Forces store to shared L1 cache line
    barrier.wait()                                   # ---- timed region ends
    del arr
    shm.close()


def measure(shm, p, stride, warmup, runs):
    iters = M_TOTAL // p

    def make():
        shm.buf[:SHM_BYTES] = bytes(SHM_BYTES)
        args = [(shm.name, (w * stride) % SLOTS, iters) for w in range(p)]
        return fs_worker, args

    st, samples = bench_procs(make, warmup, runs)
    mops = (iters * p) / (st.median_ms * 1e-3) / 1e6
    return {"p": p, "stride": stride, "stride_bytes": stride * 8,
            "min_ms": st.min_ms, "median_ms": st.median_ms, "iqr_ms": st.iqr_ms,
            "mops": mops, "samples": samples}


def karp_flatt(S, p):
    return (1.0 / S - 1.0 / p) / (1.0 - 1.0 / p)


def main():
    ap = common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    header("PHASE 1 -- False Sharing Stride Sweep & Karp-Flatt Diagnostics")
    pphys = resolve_pphys(args)
    plog = mp.cpu_count()
    line = get_cache_info()["line"] or 64
    print(f"P_phys={pphys}  P_log={plog}  cache line={line} B  M={M_TOTAL:,}  "
          f"warmup={args.warmup} runs={args.runs}")

    shm = shared_memory.SharedMemory(create=True, size=SHM_BYTES)
    try:
        # ---------------- 1. stride sweep at p = P_phys ----------------
        print(f"\n[1] Stride sweep, p = P_phys = {pphys}")
        print(f"{'stride':>6} {'bytes':>6} {'median_ms':>11} {'IQR_ms':>9} {'min_ms':>10} {'Mops/s':>9}")
        sweep = []
        for s in STRIDES:
            r = measure(shm, pphys, s, args.warmup, args.runs)
            sweep.append(r)
            print(f"{s:>6} {s*8:>6} {r['median_ms']:>11.2f} {r['iqr_ms']:>9.2f} "
                  f"{r['min_ms']:>10.2f} {r['mops']:>9.2f}", flush=True)
        # largest relative drop between consecutive strides
        drops = []
        for a, b in zip(sweep, sweep[1:]):
            drops.append(((a["median_ms"] - b["median_ms"]) / a["median_ms"], b["stride_bytes"]))
        best_drop = max(drops)
        print(f"Sharpest drop in median time: {best_drop[0]*100:.1f}% when moving to "
              f"{best_drop[1]} B offset (cache line = {line} B)")
        t_s1_pphys = sweep[0]["median_ms"]
        t_s16_pphys = sweep[STRIDES.index(16)]["median_ms"]

        # ---------------- 2. process scaling, stride 1 and 16 ----------------
        ps = pow2_sweep(plog)
        scaling = {}
        for s in (1, 16):
            print(f"\n[2] Process scaling, stride = {s} ({s*8} B)")
            print(f"{'p':>3} {'median_ms':>11} {'IQR_ms':>9} {'Mops/s':>9} {'S_p':>7} {'E_p':>7} {'e(p)':>8}")
            rows = []
            t1 = None
            for p in ps:
                if p == pphys:
                    r = next(x for x in sweep if x["stride"] == s)    # reuse sweep data
                else:
                    r = measure(shm, p, s, args.warmup, args.runs)
                if p == 1:
                    t1 = r["median_ms"]
                S = t1 / r["median_ms"]
                E = S / p
                e = karp_flatt(S, p) if p >= 2 else None
                r.update({"S": S, "E": E, "e": e})
                rows.append(r)
                es = f"{e:>8.4f}" if e is not None else f"{'-':>8}"
                print(f"{p:>3} {r['median_ms']:>11.2f} {r['iqr_ms']:>9.2f} {r['mops']:>9.2f} "
                      f"{S:>7.3f} {E:>7.3f} {es}", flush=True)
            scaling[s] = rows

        # ---------------- 3. analysis summary ----------------
        print("\n[3] Karp-Flatt e(p): stride=1 vs stride=16")
        print(f"{'p':>3} {'e(p) s=1':>10} {'e(p) s=16':>10} {'delta':>8}")
        for a, b in zip(scaling[1], scaling[16]):
            if a["e"] is None:
                continue
            print(f"{a['p']:>3} {a['e']:>10.4f} {b['e']:>10.4f} {a['e'] - b['e']:>8.4f}")
        print(f"\nFalse-sharing penalty at p=P_phys: T(stride=1)/T(stride=16) = "
              f"{t_s1_pphys / t_s16_pphys:.3f}x")

        tag = emit_verify_tag("PHASE1", STUDENT_ID, t_s1_pphys, t_s16_pphys)
        print("\n" + tag)
        for rows in [sweep] + list(scaling.values()):
            for r in rows:
                r.pop("samples", None)
        save_json("phase1.json", {"P_phys": pphys, "P_log": plog, "sweep": sweep,
                                  "scaling": {str(k): v for k, v in scaling.items()},
                                  "verify": tag})
    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()
