"""
task0_cal.py -- Phase 0: Hardware Topology Audit & Harness Calibration
Student: Ainat Yermurat (230103161)

* prints the OS-native topology command output + the baseline table values
* queries mp.cpu_count()
* measures average time.perf_counter_ns() call latency over 2 x 10^6 clock reads
* emits VERIFY_HASH[PHASE0] (student_seed = Student ID, metrics = P_log, latency_ns)
"""
import multiprocessing as mp
import platform
import time

from harness import (STUDENT_ID, emit_verify_tag, header, cpu_model, get_physical_cores,
                     get_cache_info, get_perf_cores, fmt_bytes, save_json, _run)

N_READS = 2 * 10 ** 6


def timer_latency_ns(n: int = N_READS) -> float:
    pc = time.perf_counter_ns
    t0 = pc()
    for _ in range(n):
        pc()
    t1 = pc()
    return (t1 - t0) / n


def loop_overhead_ns(n: int = N_READS) -> float:
    pc = time.perf_counter_ns
    t0 = pc()
    for _ in range(n):
        pass
    return (pc() - t0) / n


def timer_resolution_ns() -> int:
    pc = time.perf_counter_ns
    best = 10 ** 9
    for _ in range(200000):
        a = pc()
        b = pc()
        while b == a:
            b = pc()
        best = min(best, b - a)
    return best


def topology_dump():
    system = platform.system()
    if system == "Linux":
        cmd = ('lscpu | grep -E "Model name|Socket|Core|Thread|L1d|L2|L3"; '
               'echo "LEVEL1_DCACHE_LINESIZE: $(getconf LEVEL1_DCACHE_LINESIZE)"')
    elif system == "Darwin":
        cmd = ("sysctl -n machdep.cpu.brand_string; sysctl hw.physicalcpu hw.logicalcpu "
               "hw.cachelinesize hw.l1dcachesize hw.l2cachesize hw.l3cachesize")
    elif system == "Windows":
        cmd = ('powershell -NoProfile -Command "Get-CimInstance Win32_Processor | Select Name, '
               'NumberOfCores, NumberOfLogicalProcessors, L2CacheSize, L3CacheSize | Format-List"')
    else:
        return "(unknown OS)"
    print(f"$ {cmd}")
    return _run(cmd)


def main():
    header("PHASE 0 -- Hardware Topology Audit & Harness Calibration")
    print(topology_dump())
    print("-" * 76)

    plog = mp.cpu_count()
    pphys = get_physical_cores()
    cache = get_cache_info()
    perf = get_perf_cores()
    pyos = f"Python {platform.python_version()} / {platform.system()} {platform.release()}"

    print("Hardware baseline table")
    print(f"  CPU Model String        : {cpu_model()}")
    print(f"  Physical Cores (P_phys) : {pphys}")
    print(f"  Logical Threads (P_log) : {plog}")
    if perf:
        print(f"  Performance cores       : {perf}")
    print(f"  Cache Line Size (L)     : {cache['line']} B")
    print(f"  L1 Data Cache (per core): {fmt_bytes(cache['l1d'])}")
    print(f"  L2 Cache (per instance) : {fmt_bytes(cache['l2'])}")
    print(f"  L3 Cache (shared)       : {fmt_bytes(cache['l3'])}")
    print(f"  Python Version & OS     : {pyos}")
    print(f"  (cache source: {cache['source']})")
    print("-" * 76)

    timer_latency_ns(200000)                                 # warm-up
    lat = sorted(timer_latency_ns() for _ in range(5))
    ovh = sorted(loop_overhead_ns() for _ in range(5))
    lat_med, ovh_med = lat[2], ovh[2]
    res = timer_resolution_ns()
    print(f"perf_counter_ns() avg latency over 2x10^6 reads : {lat_med:.2f} ns/call (median of 5)")
    print(f"  empty for-loop overhead                        : {ovh_med:.2f} ns/iter")
    print(f"  net clock-read cost                            : {lat_med - ovh_med:.2f} ns")
    print(f"perf_counter_ns() observed resolution            : {res} ns")
    print("-" * 76)

    tag = emit_verify_tag("PHASE0", STUDENT_ID, float(plog), lat_med)
    print(tag)
    save_json("phase0.json", {"cpu_model": cpu_model(), "P_phys": pphys, "P_log": plog,
                              "perf_cores": perf, "cache": cache, "python_os": pyos,
                              "timer_latency_ns": lat_med, "loop_overhead_ns": ovh_med,
                              "timer_resolution_ns": res, "verify": tag})


if __name__ == "__main__":
    main()
