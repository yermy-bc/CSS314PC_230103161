"""
task2_scaling.py -- Phase 2: Strong Scaling, Weak Scaling & Architectural Ceilings
Student: Ainat Yermurat (230103161)

Per trial over N elements:
  Stage A (strictly serial, parent): first 5% of elements (0.05N),
          12 transcendental ops/element (6 x math.sin + 6 x math.sqrt).
  Stage B (embarrassingly parallel): all N elements split evenly over a persistent
          mp.Pool(processes=p) created OUTSIDE the timed region,
          35 multiply-add / trig ops per element, zero shared writes.
"""
import argparse
import math
import multiprocessing as mp
import time

from harness import (STUDENT_ID, emit_verify_tag, header, run_benchmark, save_json,
                     common_args, resolve_pphys, get_perf_cores)

N_STRONG = 1_600_000
N0_WEAK = 250_000
STAGE_A_FRAC = 0.05


def stage_a(n_a: int) -> float:
    """Serial reduction/prefix filter: 12 transcendental ops per element."""
    sin, sqrt = math.sin, math.sqrt
    acc = 0.0
    for i in range(n_a):
        x = i * 1e-6 + 1.0
        x = sin(x); x = sqrt(x * x + 1.0)
        x = sin(x); x = sqrt(x * x + 1.0)
        x = sin(x); x = sqrt(x * x + 1.0)
        x = sin(x); x = sqrt(x * x + 1.0)
        x = sin(x); x = sqrt(x * x + 1.0)
        x = sin(x); x = sqrt(x * x + 1.0)
        if x > 0.0:                      # prefix filter
            acc += x
    return acc


def stage_b_chunk(bounds) -> float:
    """35 ops/element: 7 x (FMA, sin, FMA, FMA, cos). Returns a private partial sum."""
    start, end = bounds
    sin, cos = math.sin, math.cos
    a, b, c = 0.999999, 1.000001, 1e-7
    acc = 0.0
    for i in range(start, end):
        x = i * 1e-6
        y = x
        z = 0.5
        for _ in range(7):
            y = y * a + c        # FMA
            y = sin(y)           # trig
            z = z * b + y        # FMA
            y = y * a + x        # FMA
            z = cos(z)           # trig
        acc += y + z
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
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    ap = common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    header("PHASE 2 -- Strong Scaling, Weak Scaling & Architectural Ceilings")
    pphys = resolve_pphys(args)
    plog = mp.cpu_count()
    perf = get_perf_cores()
    print(f"P_phys={pphys}  P_log={plog}  perf_cores={perf}  "
          f"warmup_runs={args.warmup} timed_runs={args.runs}")

    strong, weak = [], []
    for p in range(1, plog + 1):
        with mp.Pool(processes=p) as pool:                       # outside timed region
            pool.map(stage_b_chunk, chunks(1000 * p, p), chunksize=1)   # spin-up
            la = []
            st = run_benchmark(lambda: trial(pool, N_STRONG, p, la), args.warmup, args.runs)
            strong.append({"p": p, "median_ms": st.median_ms, "iqr_ms": st.iqr_ms,
                           "min_ms": st.min_ms, "max_ms": st.max_ms,
                           "stageA_ms": median(la[args.warmup:])})
            nw = p * N0_WEAK
            lw = []
            wt = run_benchmark(lambda: trial(pool, nw, p, lw), args.warmup, args.runs)
            weak.append({"p": p, "N": nw, "median_ms": wt.median_ms, "iqr_ms": wt.iqr_ms,
                         "min_ms": wt.min_ms, "max_ms": wt.max_ms,
                         "stageA_ms": median(lw[args.warmup:])})
        print(f"  p={p:>2}: T_strong={st.median_ms:9.1f} ms   T_weak={wt.median_ms:9.1f} ms", flush=True)

    # ---------------- Experiment 2A: strong scaling ----------------
    T1 = strong[0]["median_ms"]
    s_frac = strong[0]["stageA_ms"] / T1
    S_inf = 1.0 / s_frac
    amd_plog = 1.0 / (s_frac + (1 - s_frac) / plog)
    print(f"\nExperiment 2A -- Strong scaling, N = {N_STRONG:,}")
    print(f"s = T_StageA(1)/T_1 = {strong[0]['stageA_ms']:.2f}/{T1:.2f} = {s_frac:.5f}   "
          f"S_inf = 1/s = {S_inf:.2f}   Amdahl S({plog}) = {amd_plog:.3f}")
    print(f"{'p':>3} {'T_p ms':>10} {'IQR':>8} {'S_p':>7} {'E_p':>7} {'Amdahl':>8} {'dS':>7}")
    prevS = 0.0
    for r in strong:
        p = r["p"]
        S = T1 / r["median_ms"]
        E = S / p
        amd = 1.0 / (s_frac + (1 - s_frac) / p)
        r.update({"S": S, "E": E, "amdahl": amd, "dS": S - prevS})
        print(f"{p:>3} {r['median_ms']:>10.1f} {r['iqr_ms']:>8.1f} {S:>7.3f} {E:>7.3f} {amd:>8.3f} {S - prevS:>7.3f}")
        prevS = S
    S_plog = strong[-1]["S"]
    print(f"Observed S_strong({plog}) = {S_plog:.3f}  vs Amdahl S({plog}) = {amd_plog:.3f} "
          f"({S_plog / amd_plog * 100:.1f}%)  vs S_inf = {S_inf:.2f} ({S_plog / S_inf * 100:.1f}%)")

    # ---------------- Experiment 2B: weak scaling ----------------
    Tw1 = weak[0]["median_ms"]
    print(f"\nExperiment 2B -- Weak scaling, N0 = {N0_WEAK:,}, N(p) = p*N0")
    print(f"{'p':>3} {'N':>11} {'T_p ms':>10} {'IQR':>8} {'E_weak':>7} {'alpha':>7} {'S_Gust':>8} {'p*E_weak':>9}")
    for r in weak:
        p = r["p"]
        Ew = Tw1 / r["median_ms"]
        alpha = r["stageA_ms"] / r["median_ms"]
        Sg = p - alpha * (p - 1)
        r.update({"E_weak": Ew, "alpha": alpha, "S_gust": Sg, "scaled_measured": p * Ew})
        print(f"{p:>3} {r['N']:>11,} {r['median_ms']:>10.1f} {r['iqr_ms']:>8.1f} {Ew:>7.3f} "
              f"{alpha:>7.4f} {Sg:>8.3f} {p * Ew:>9.3f}")
    Sg_plog = weak[-1]["S_gust"]

    # ---------------- inflection analysis ----------------
    print("\nArchitectural inflection (marginal speedup per added worker)")
    def dS(p):
        return strong[p - 1]["dS"]
    infl = {}
    if pphys < plog:
        b, a = dS(pphys), dS(pphys + 1)
        drop_pct = (b - a) / b * 100 if b else float("nan")
        avg_phys = (strong[pphys - 1]["S"] - 1.0) / (pphys - 1) if pphys > 1 else float("nan")
        avg_smt = (strong[-1]["S"] - strong[pphys - 1]["S"]) / (plog - pphys)
        avg_drop = (avg_phys - avg_smt) / avg_phys * 100
        print(f"dS at p={pphys} (last physical core)        : {b:+.3f}")
        print(f"dS at p={pphys + 1} (first SMT sibling)        : {a:+.3f}")
        print(f"Drop in marginal speedup P_phys -> P_phys+1: {drop_pct:.1f}%")
        print(f"Mean dS per worker, p=1..{pphys} (physical)   : {avg_phys:.3f}")
        print(f"Mean dS per worker, p={pphys}..{plog} (SMT)      : {avg_smt:.3f}")
        print(f"Drop in mean marginal speedup per worker   : {avg_drop:.1f}%")
        print(f"SMT yield S({plog})/S({pphys})                 : {strong[-1]['S'] / strong[pphys - 1]['S']:.3f}")
        infl = {"dS_pphys": b, "dS_pphys1": a, "drop_pct": drop_pct, "avg_phys": avg_phys,
                "avg_smt": avg_smt, "avg_drop_pct": avg_drop}
    else:
        print(f"P_log == P_phys ({plog}): no SMT; see P/E-core boundary below if reported.")
    if perf and perf < plog:
        print(f"P-core -> E-core boundary at p={perf}: dS {dS(perf):+.3f} -> {dS(perf + 1):+.3f}")

    tag = emit_verify_tag("PHASE2", STUDENT_ID, S_plog, Sg_plog)
    print("\n" + tag)
    save_json("phase2.json", {"P_phys": pphys, "P_log": plog, "s": s_frac, "S_inf": S_inf,
                              "amdahl_plog": amd_plog, "strong": strong, "weak": weak,
                              "inflection": infl, "verify": tag})


if __name__ == "__main__":
    main()
