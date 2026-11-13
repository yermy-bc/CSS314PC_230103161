"""
task2_scaling.py -- Phase 2: Strong Scaling vs. Weak Scaling
Student: Ainat Yermurat (230103161)

Two-stage pipeline per trial:
  Stage A: serial, in the parent, over the first 5% of elements (0.05*N),
           10 math.sin / math.sqrt ops per element.
  Stage B: all N elements partitioned into p contiguous chunks on a persistent
           mp.Pool(processes=p) created OUTSIDE the timed loop,
           30 FMA/trig ops per element, zero shared writes (each task returns a float).
"""
import argparse
import math
import multiprocessing as mp
import time

from harness import (STUDENT_ID, emit_verify_tag, header, run_benchmark, save_json,
                     common_args, resolve_pphys, get_perf_cores)

N_STRONG = 1_200_000
N0_WEAK = 200_000
STAGE_A_FRAC = 0.05


def stage_a(n_a: int) -> float:
    """10 ops/element: 5 x sin + 5 x sqrt (serial part)."""
    sin, sqrt = math.sin, math.sqrt
    acc = 0.0
    for i in range(n_a):
        x = i * 1e-6 + 1.0
        x = sin(x); x = sqrt(x * x + 1.0)
        x = sin(x); x = sqrt(x * x + 1.0)
        x = sin(x); x = sqrt(x * x + 1.0)
        x = sin(x); x = sqrt(x * x + 1.0)
        x = sin(x); x = sqrt(x * x + 1.0)
        acc += x
    return acc


def stage_b_chunk(bounds) -> float:
    """30 ops/element: 10 x (FMA, sin, FMA). Returns a private partial sum."""
    start, end = bounds
    sin = math.sin
    a, c = 0.999999, 1e-7
    acc = 0.0
    for i in range(start, end):
        x = i * 1e-6
        y = x
        for _ in range(10):
            y = y * a + c      # FMA
            y = sin(y)         # trig
            y = y * a + x      # FMA
        acc += y
    return acc


def chunks(n: int, p: int):
    q, r = divmod(n, p)
    out, s = [], 0
    for w in range(p):
        e = s + q + (1 if w < r else 0)
        out.append((s, e))
        s = e
    return out


def trial(pool, n, p, stage_a_log):
    t0 = time.perf_counter_ns()
    stage_a(int(STAGE_A_FRAC * n))
    stage_a_log.append((time.perf_counter_ns() - t0) / 1e6)
    pool.map(stage_b_chunk, chunks(n, p), chunksize=1)


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def main():
    ap = common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    header("PHASE 2 -- Strong Scaling vs. Weak Scaling")
    pphys = resolve_pphys(args)
    plog = mp.cpu_count()
    perf = get_perf_cores()
    print(f"P_phys={pphys}  P_log={plog}  perf_cores={perf}  warmup={args.warmup} runs={args.runs}")

    strong, weak = [], []
    for p in range(1, plog + 1):
        with mp.Pool(processes=p) as pool:
            pool.map(stage_b_chunk, chunks(1000 * p, p), chunksize=1)   # spin up workers
            la = []
            st = run_benchmark(lambda: trial(pool, N_STRONG, p, la), args.warmup, args.runs)
            strong.append({"p": p, "median_ms": st.median_ms, "iqr_ms": st.iqr_ms,
                           "min_ms": st.min_ms, "stageA_ms": median(la[args.warmup:])})
            nw = p * N0_WEAK
            lw = []
            wt = run_benchmark(lambda: trial(pool, nw, p, lw), args.warmup, args.runs)
            weak.append({"p": p, "N": nw, "median_ms": wt.median_ms, "iqr_ms": wt.iqr_ms,
                         "min_ms": wt.min_ms, "stageA_ms": median(lw[args.warmup:])})
        print(f"  p={p:>2}: T_strong={st.median_ms:9.1f} ms   T_weak={wt.median_ms:9.1f} ms", flush=True)

    # ---------------- strong scaling ----------------
    T1 = strong[0]["median_ms"]
    s_frac = strong[0]["stageA_ms"] / T1
    S_inf = 1.0 / s_frac
    print(f"\n[1] Strong scaling  N = {N_STRONG:,}   s = T_A(1)/T_1 = "
          f"{strong[0]['stageA_ms']:.1f}/{T1:.1f} = {s_frac:.4f}   Amdahl cap S_inf = 1/s = {S_inf:.2f}")
    print(f"{'p':>3} {'T_p ms':>10} {'IQR':>8} {'S_p':>7} {'E_p':>7} {'Amdahl S(p)':>12} {'dS':>7}")
    prevS = 0.0
    for r in strong:
        p = r["p"]
        S = T1 / r["median_ms"]
        E = S / p
        amd = 1.0 / (s_frac + (1 - s_frac) / p)
        r.update({"S": S, "E": E, "amdahl": amd, "dS": S - prevS})
        print(f"{p:>3} {r['median_ms']:>10.1f} {r['iqr_ms']:>8.1f} {S:>7.3f} {E:>7.3f} {amd:>12.3f} {S - prevS:>7.3f}")
        prevS = S
    S_plog = strong[-1]["S"]
    print(f"S_strong(P_log={plog}) = {S_plog:.3f}  vs Amdahl cap {S_inf:.2f} "
          f"({S_plog / S_inf * 100:.1f}% of cap)")

    # ---------------- weak scaling ----------------
    Tw1 = weak[0]["median_ms"]
    print(f"\n[2] Weak scaling  N0 = {N0_WEAK:,}  N(p) = p*N0")
    print(f"{'p':>3} {'N':>10} {'T_p ms':>10} {'IQR':>8} {'E_weak':>7} {'alpha':>7} {'S_Gust':>8}")
    for r in weak:
        p = r["p"]
        Ew = Tw1 / r["median_ms"]
        alpha = r["stageA_ms"] / r["median_ms"]
        Sg = p - alpha * (p - 1)
        r.update({"E_weak": Ew, "alpha": alpha, "S_gust": Sg})
        print(f"{p:>3} {r['N']:>10,} {r['median_ms']:>10.1f} {r['iqr_ms']:>8.1f} {Ew:>7.3f} {alpha:>7.4f} {Sg:>8.3f}")
    Sg_plog = weak[-1]["S_gust"]

    # ---------------- architectural inflection ----------------
    print("\n[3] Architectural inflection")
    def marg(p):
        return strong[p - 1]["S"] - strong[p - 2]["S"] if 2 <= p <= plog else None
    if pphys < plog:
        before = marg(pphys)
        after = marg(pphys + 1)
        print(f"Marginal speedup dS at p={pphys} (last physical core): {before:.3f}")
        print(f"Marginal speedup dS at p={pphys + 1} (first SMT/E-core): {after:.3f}")
        if before:
            print(f"Drop across P_phys -> P_phys+1: {before - after:.3f} "
                  f"({(1 - after / before) * 100:.1f}% lower)")
    else:
        print(f"P_log == P_phys ({plog}): no SMT threads, p = P_phys+1 is not available.")
    if perf and perf < plog:
        b, a = marg(perf), marg(perf + 1)
        print(f"P-core -> E-core boundary (perf cores = {perf}): dS {b:.3f} -> {a:.3f}")

    tag = emit_verify_tag("PHASE2", STUDENT_ID, S_plog, Sg_plog)
    print("\n" + tag)
    save_json("phase2.json", {"P_phys": pphys, "P_log": plog, "s": s_frac, "S_inf": S_inf,
                              "strong": strong, "weak": weak, "verify": tag})


if __name__ == "__main__":
    main()
