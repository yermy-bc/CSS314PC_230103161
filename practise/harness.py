"""
harness.py -- Minimal Timing and Verification Harness (Listing 1)
Performance Evaluation and Optimization in Parallel Computing -- Minimal Python Lab

Student : Ainat Yermurat
ID      : 230103161

Lines 1-33 of Listing 1 are reproduced verbatim (BenchStats, fnv1a_hash,
run_benchmark, emit_verify_tag). Everything below the marked line is
shared helper code used by task0..task4 (hardware detection, barrier-timed
process launch, sample summarising, JSON saving).
"""
from dataclasses import dataclass
import multiprocessing as mp
import time
from typing import Callable


@dataclass
class BenchStats:
    min_ms: float
    median_ms: float
    iqr_ms: float


def fnv1a_hash(s: str) -> int:
    h = 14695981039346656037
    for b in s.encode("utf-8"):
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def run_benchmark(fn: Callable[[], None], warmup: int = 2, runs: int = 7) -> BenchStats:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    samples.sort()
    med = samples[runs // 2]
    return BenchStats(samples[0], med, samples[(3 * runs) // 4] - samples[runs // 4])


def emit_verify_tag(phase: str, seed: int, m1: float, m2: float) -> str:
    hw = mp.cpu_count()
    sig = fnv1a_hash(f"{phase}|{seed}|{hw}|{m1:.4f}|{m2:.4f}")
    return f"VERIFY_HASH[{phase}]: {sig:016X} (P_log={hw})"


# ===================================================================
# ----------------- shared helpers (not in Listing 1) ---------------
# ===================================================================
import json
import os
import platform
import re
import subprocess
import sys
from multiprocessing import shared_memory

STUDENT_ID = 230103161
STUDENT_NAME = "Ainat Yermurat"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def summarize(samples):
    """Same statistics as run_benchmark(), for samples measured elsewhere."""
    s = sorted(samples)
    n = len(s)
    return BenchStats(s[0], s[n // 2], s[(3 * n) // 4] - s[n // 4])


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20,
                              shell=isinstance(cmd, str)).stdout.strip()
    except Exception:
        return ""


def get_physical_cores() -> int:
    """Best-effort physical core count (P_phys). Override with env LAB_PPHYS."""
    env = os.environ.get("LAB_PPHYS")
    if env:
        return int(env)
    try:
        import psutil  # type: ignore
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    system = platform.system()
    try:
        if system == "Linux":
            cores = set()
            phys = core = None
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("physical id"):
                        phys = line.split(":")[1].strip()
                    elif line.startswith("core id"):
                        core = line.split(":")[1].strip()
                    elif not line.strip():
                        if core is not None:
                            cores.add((phys, core))
                        phys = core = None
            if core is not None:
                cores.add((phys, core))
            if cores:
                return len(cores)
        elif system == "Darwin":
            out = _run(["sysctl", "-n", "hw.physicalcpu"])
            if out.isdigit():
                return int(out)
        elif system == "Windows":
            out = _run(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"])
            if out.strip().isdigit():
                return int(out.strip())
    except Exception:
        pass
    return mp.cpu_count()


def get_perf_cores():
    """Apple Silicon / hybrid: number of performance cores if the OS exposes it."""
    if platform.system() == "Darwin":
        out = _run(["sysctl", "-n", "hw.perflevel0.physicalcpu"])
        if out.isdigit():
            return int(out)
    return None


def get_cache_info():
    """Best-effort cache sizes in bytes: dict(line, l1d, l2, l3). Missing -> None."""
    info = {"line": None, "l1d": None, "l2": None, "l3": None}
    system = platform.system()
    try:
        if system == "Linux":
            base = "/sys/devices/system/cpu/cpu0/cache"
            for idx in sorted(os.listdir(base)):
                if not idx.startswith("index"):
                    continue
                d = os.path.join(base, idx)
                lvl = open(os.path.join(d, "level")).read().strip()
                typ = open(os.path.join(d, "type")).read().strip()
                size = open(os.path.join(d, "size")).read().strip()
                m = re.match(r"(\d+)([KMG]?)", size)
                mult = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}[m.group(2)]
                val = int(m.group(1)) * mult
                if lvl == "1" and typ in ("Data", "Unified"):
                    info["l1d"] = val
                    info["line"] = int(open(os.path.join(d, "coherency_line_size")).read())
                elif lvl == "2":
                    info["l2"] = val
                elif lvl == "3":
                    info["l3"] = val
        elif system == "Darwin":
            def sc(k):
                o = _run(["sysctl", "-n", k])
                return int(o) if o.isdigit() else None
            info["line"] = sc("hw.cachelinesize")
            info["l1d"] = sc("hw.perflevel0.l1dcachesize") or sc("hw.l1dcachesize")
            info["l2"] = sc("hw.perflevel0.l2cachesize") or sc("hw.l2cachesize")
            info["l3"] = sc("hw.l3cachesize")
        elif system == "Windows":
            out = _run(["powershell", "-NoProfile", "-Command",
                        "Get-CimInstance Win32_Processor | Select-Object -First 1 L2CacheSize,L3CacheSize | ConvertTo-Json"])
            if out:
                d = json.loads(out)
                info["l2"] = (d.get("L2CacheSize") or 0) * 1024 or None
                info["l3"] = (d.get("L3CacheSize") or 0) * 1024 or None
            info["line"] = 64
    except Exception:
        pass
    return info


def cpu_model() -> str:
    system = platform.system()
    if system == "Linux":
        try:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    elif system == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out
    elif system == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"])
        if out:
            return out
    return platform.processor() or "unknown"


def pow2_sweep(plog: int):
    """{1, 2, 4, ..., P_log}  (P_log appended if it is not a power of two)."""
    ps, p = [], 1
    while p < plog:
        ps.append(p)
        p *= 2
    ps.append(plog)
    return ps


def attach_shm(name: str) -> shared_memory.SharedMemory:
    """Attach to an existing block without letting the child's resource tracker own it."""
    if sys.version_info >= (3, 13):
        return shared_memory.SharedMemory(name=name, track=False)
    return shared_memory.SharedMemory(name=name)


def run_procs_timed(target, args_list, timeout: float = 1800.0) -> float:
    """
    Launch one mp.Process per entry in args_list running target(*args, barrier).
    Each worker must call barrier.wait() once when ready (after attaching shared
    memory) and once after finishing its work. The parent times the interval
    between the two barrier crossings, so process spawn / teardown cost is
    excluded. Returns elapsed wall time in ms.
    """
    n = len(args_list)
    barrier = mp.Barrier(n + 1)
    procs = [mp.Process(target=target, args=tuple(a) + (barrier,)) for a in args_list]
    for pr in procs:
        pr.start()
    barrier.wait(timeout=timeout)          # all workers ready
    t0 = time.perf_counter_ns()
    barrier.wait(timeout=timeout)          # all workers done
    t1 = time.perf_counter_ns()
    for pr in procs:
        pr.join()
        if pr.exitcode != 0:
            raise RuntimeError(f"worker exited with code {pr.exitcode}")
    return (t1 - t0) / 1e6


def bench_procs(make_args, warmup: int = 2, runs: int = 7):
    """Barrier-timed equivalent of run_benchmark() for process-launch kernels.
    make_args() -> (target, args_list). Returns (BenchStats, samples)."""
    for _ in range(warmup):
        t, a = make_args()
        run_procs_timed(t, a)
    samples = []
    for _ in range(runs):
        t, a = make_args()
        samples.append(run_procs_timed(t, a))
    return summarize(samples), samples


def save_json(name: str, data) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def header(title: str):
    print("=" * 72)
    print(f"{title}")
    print(f"Student: {STUDENT_NAME}  ID: {STUDENT_ID}")
    print(f"Python {platform.python_version()} | {platform.system()} {platform.release()} | "
          f"start method: {mp.get_start_method()}")
    print("=" * 72)


def common_args(parser, warmup=2, runs=7):
    parser.add_argument("--warmup", type=int, default=warmup)
    parser.add_argument("--runs", type=int, default=runs)
    parser.add_argument("--pphys", type=int, default=None,
                        help="override detected physical core count")
    return parser


def resolve_pphys(args) -> int:
    return args.pphys if args.pphys else get_physical_cores()
