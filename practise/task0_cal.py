"""
task0_cal.py -- Phase 0: Hardware Topology & Timer Calibration
Student: Ainat Yermurat (230103161)

* queries mp.cpu_count()
* measures average latency of time.perf_counter_ns() over 10^6 consecutive calls
* prints the OS-specific topology command output (for the Phase 0 table)
* emits VERIFY_HASH[PHASE0] with the Student ID as seed
"""
import multiprocessing as mp
import platform
import time

from harness import (STUDENT_ID, emit_verify_tag, header, cpu_model, get_physical_cores,
                     get_cache_info, get_perf_cores, save_json, _run)

N_CALLS = 10 ** 6


def timer_latency_ns(n: int = N_CALLS) -> float:
    pc = time.perf_counter_ns
    t0 = pc()
    for _ in range(n):
        pc()
    t1 = pc()
    return (t1 - t0) / n


def loop_overhead_ns(n: int = N_CALLS) -> float:
    pc = time.perf_counter_ns
    t0 = pc()
    for _ in range(n):
        pass
    t1 = pc()
    return (t1 - t0) / n


def timer_resolution_ns() -> int:
    """Smallest non-zero difference between two consecutive perf_counter_ns reads."""
    pc = time.perf_counter_ns
    best = 10 ** 9
    for _ in range(200000):
        a = pc()
        b = pc()
        while b == a:
            b = pc()
        if b - a < best:
            best = b - a
    return best


def topology_dump():
    system = platform.system()
    if system == "Linux":
        cmd = ('lscpu | grep -E "Model name|Core|Thread|L1d|L2|L3"; '
               'echo "LEVEL1_DCACHE_LINESIZE: $(getconf LEVEL1_DCACHE_LINESIZE)"')
    elif system == "Darwin":
        cmd = ("sysctl -n machdep.cpu.brand_string; sysctl hw.physicalcpu hw.logicalcpu "
               "hw.cachelinesize hw.l1dcachesize hw.l2cachesize hw.l3cachesize; "
               "sysctl hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu 2>/dev/null")
    elif system == "Windows":
        cmd = ('powershell -NoProfile -Command "Get-CimInstance Win32_Processor | Select Name, '
               'NumberOfCores, NumberOfLogicalProcessors, L2CacheSize, L3CacheSize | Format-List"')
    else:
        return "(unknown OS)"
    print(f"$ {cmd}")
    return _run(cmd)


def main():
    header("PHASE 0 -- Hardware Topology & Minimal Telemetry Harness")
    print(topology_dump())
    print("-" * 72)

    plog = mp.cpu_count()
    pphys = get_physical_cores()
    cache = get_cache_info()
    perf = get_perf_cores()

    def kb(x):
        return "n/a" if not x else (f"{x // 1024} KiB" if x < 1024 ** 2 else f"{x / 1024 ** 2:.1f} MiB")

    print(f"CPU model            : {cpu_model()}")
    print(f"P_phys (physical)    : {pphys}")
    print(f"P_log  (mp.cpu_count): {plog}")
    if perf:
        print(f"Performance cores    : {perf}")
    print(f"Cache line (L bytes) : {cache['line']}")
    print(f"L1d / L2 / L3 (per-core-or-shared as OS reports): "
          f"{kb(cache['l1d'])} / {kb(cache['l2'])} / {kb(cache['l3'])}")
    print("-" * 72)

    # warm the timer and the loop once, then take the median of 5 calibrations
    timer_latency_ns(100000)
    lat = sorted(timer_latency_ns() for _ in range(5))
    ovh = sorted(loop_overhead_ns() for _ in range(5))
    lat_med, ovh_med = lat[2], ovh[2]
    res = timer_resolution_ns()

    print(f"perf_counter_ns() avg latency over 10^6 calls : {lat_med:.2f} ns/call (median of 5)")
    print(f"  (empty for-loop overhead                    : {ovh_med:.2f} ns/iter)")
    print(f"  (net call cost                              : {lat_med - ovh_med:.2f} ns)")
    print(f"perf_counter_ns() observed resolution          : {res} ns")
    print("-" * 72)

    tag = emit_verify_tag("PHASE0", STUDENT_ID, float(plog), lat_med)
    print(tag)

    save_json("phase0.json", {
        "cpu_model": cpu_model(), "P_phys": pphys, "P_log": plog, "perf_cores": perf,
        "cache": cache, "timer_latency_ns": lat_med, "loop_overhead_ns": ovh_med,
        "timer_resolution_ns": res, "verify": tag,
    })


if __name__ == "__main__":
    main()
