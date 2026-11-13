"""
harness.py -- Zero-Dependency Python 3 Telemetry & Verification Framework (Listing 1)
Performance Evaluation and Optimization in Parallel Computing -- Solo Python Systems Lab

Student : Ainat Yermurat
ID      : 230103161

The first block (BenchStats, fnv1a_hash, run_benchmark, emit_verify_tag) is
Listing 1 reproduced verbatim. Everything below the marked line is shared
helper code used by task0..task4: hardware detection (incl. exact Windows cache
sizes via GetLogicalProcessorInformation), a persistent-worker ProcGroup so
run_benchmark() never times process spawn, and JSON/result helpers.
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
    max_ms: float


def fnv1a_hash(s: str) -> int:
    h = 14695981039346656037
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def run_benchmark(fn: Callable[[], None], warmup_runs: int = 3, timed_runs: int = 9) -> BenchStats:
    for _ in range(warmup_runs):
        fn()
    samples = []
    for _ in range(timed_runs):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e6)
    samples.sort()
    mid = timed_runs // 2
    med = samples[mid] if timed_runs % 2 == 1 else 0.5 * (samples[mid - 1] + samples[mid])
    q25 = samples[timed_runs // 4]
    q75 = samples[(3 * timed_runs) // 4]
    return BenchStats(samples[0], med, q75 - q25, samples[-1])


def emit_verify_tag(task_id: str, student_seed: int,
                    primary_metric: float, secondary_metric: float) -> str:
    hw = mp.cpu_count()
    payload = (
        f"{task_id}|{student_seed}|{hw}|"
        f"{primary_metric:.4f}|{secondary_metric:.4f}"
    )
    sig = fnv1a_hash(payload)
    return f"VERIFY_HASH[{task_id}]: {sig:016X} (P_log={hw})"


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
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
BARRIER_TIMEOUT = 1800.0


def summarize(samples):
    """BenchStats for samples measured outside run_benchmark (same formulas)."""
    s = sorted(samples)
    n = len(s)
    mid = n // 2
    med = s[mid] if n % 2 == 1 else 0.5 * (s[mid - 1] + s[mid])
    return BenchStats(s[0], med, s[(3 * n) // 4] - s[n // 4], s[-1])


# ------------------------- persistent workers -------------------------
def _server(target, args, start_b, done_b, stop):
    """Worker loop: wait for a start signal, run target(*args) once, report done."""
    while True:
        start_b.wait(timeout=BARRIER_TIMEOUT)
        if stop.value:
            break
        target(*args)
        done_b.wait(timeout=BARRIER_TIMEOUT)


class ProcGroup:
    """
    p persistent mp.Process workers, each running target(*args_list[t]) once per
    run() call. Processes are spawned once in __init__, so a run_benchmark(group.run)
    measurement contains only kernel time (+ ~tens of us of barrier release),
    never fork/spawn cost.
    """

    def __init__(self, target, args_list):
        n = len(args_list)
        self.start_b = mp.Barrier(n + 1)
        self.done_b = mp.Barrier(n + 1)
        self.stop = mp.Value("b", 0)
        self.procs = [mp.Process(target=_server,
                                 args=(target, tuple(a), self.start_b, self.done_b, self.stop))
                      for a in args_list]
        for pr in self.procs:
            pr.start()

    def run(self):
        self.start_b.wait(timeout=BARRIER_TIMEOUT)
        self.done_b.wait(timeout=BARRIER_TIMEOUT)

    def close(self):
        self.stop.value = 1
        try:
            self.start_b.wait(timeout=60)
        except Exception:
            pass
        for pr in self.procs:
            pr.join(timeout=60)
            if pr.is_alive():
                pr.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def attach_shm(name: str) -> shared_memory.SharedMemory:
    """Attach without letting the child's resource tracker own the block (3.13+)."""
    if sys.version_info >= (3, 13):
        return shared_memory.SharedMemory(name=name, track=False)
    return shared_memory.SharedMemory(name=name)


# ------------------------- hardware detection -------------------------
def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                              shell=isinstance(cmd, str)).stdout.strip()
    except Exception:
        return ""


def _win_slpi():
    """Windows: GetLogicalProcessorInformation -> (physical cores, {(level,type): size}, line)."""
    import ctypes
    from ctypes import wintypes

    class CACHE_DESCRIPTOR(ctypes.Structure):
        _fields_ = [("Level", ctypes.c_ubyte), ("Associativity", ctypes.c_ubyte),
                    ("LineSize", ctypes.c_ushort), ("Size", ctypes.c_uint32), ("Type", ctypes.c_int)]

    class _U(ctypes.Union):
        _fields_ = [("Flags", ctypes.c_ubyte), ("NodeNumber", ctypes.c_uint32),
                    ("Cache", CACHE_DESCRIPTOR), ("Reserved", ctypes.c_ulonglong * 2)]

    class SLPI(ctypes.Structure):
        _fields_ = [("ProcessorMask", ctypes.c_size_t), ("Relationship", ctypes.c_int), ("u", _U)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    length = wintypes.DWORD(0)
    k32.GetLogicalProcessorInformation(None, ctypes.byref(length))
    n = length.value // ctypes.sizeof(SLPI)
    buf = (SLPI * n)()
    if not k32.GetLogicalProcessorInformation(buf, ctypes.byref(length)):
        raise OSError("GetLogicalProcessorInformation failed")
    cores, caches, line = 0, {}, None
    for e in buf:
        if e.Relationship == 0:                      # RelationProcessorCore
            cores += 1
        elif e.Relationship == 2:                    # RelationCache
            c = e.u.Cache
            caches[(c.Level, c.Type)] = c.Size       # per-instance size
            if c.Level == 1:
                line = c.LineSize
    return cores, caches, line


def get_physical_cores() -> int:
    env = os.environ.get("LAB_PPHYS")
    if env:
        return int(env)
    system = platform.system()
    try:
        if system == "Windows":
            try:
                cores, _, _ = _win_slpi()
                if cores:
                    return cores
            except Exception:
                pass
            out = _run(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"])
            if out.strip().isdigit():
                return int(out.strip())
        if system == "Darwin":
            out = _run(["sysctl", "-n", "hw.physicalcpu"])
            if out.isdigit():
                return int(out)
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
    except Exception:
        pass
    try:
        import psutil  # type: ignore
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    return mp.cpu_count()


def get_perf_cores():
    if platform.system() == "Darwin":
        out = _run(["sysctl", "-n", "hw.perflevel0.physicalcpu"])
        if out.isdigit():
            return int(out)
    return None


def get_cache_info():
    """Per-instance cache sizes in bytes: line, l1d (per core), l2, l3 (shared)."""
    info = {"line": None, "l1d": None, "l2": None, "l3": None, "source": None}
    system = platform.system()
    try:
        if system == "Windows":
            try:
                _, caches, line = _win_slpi()
                info["line"] = line
                info["l1d"] = caches.get((1, 2)) or caches.get((1, 0))   # Data / Unified
                info["l2"] = caches.get((2, 0)) or caches.get((2, 2))
                info["l3"] = caches.get((3, 0)) or caches.get((3, 2))
                info["source"] = "GetLogicalProcessorInformation"
            except Exception as ex:
                out = _run(["powershell", "-NoProfile", "-Command",
                            "Get-CimInstance Win32_Processor | Select-Object -First 1 "
                            "L2CacheSize,L3CacheSize | ConvertTo-Json"])
                d = json.loads(out) if out else {}
                info["l2"] = (d.get("L2CacheSize") or 0) * 1024 or None
                info["l3"] = (d.get("L3CacheSize") or 0) * 1024 or None
                info["line"] = 64
                info["source"] = f"Win32_Processor fallback ({ex})"
        elif system == "Linux":
            base = "/sys/devices/system/cpu/cpu0/cache"
            for idx in sorted(os.listdir(base)):
                if not idx.startswith("index"):
                    continue
                d = os.path.join(base, idx)
                lvl = open(os.path.join(d, "level")).read().strip()
                typ = open(os.path.join(d, "type")).read().strip()
                size = open(os.path.join(d, "size")).read().strip()
                m = re.match(r"(\d+)([KMG]?)", size)
                val = int(m.group(1)) * {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}[m.group(2)]
                if lvl == "1" and typ in ("Data", "Unified"):
                    info["l1d"] = val
                    info["line"] = int(open(os.path.join(d, "coherency_line_size")).read())
                elif lvl == "2":
                    info["l2"] = val
                elif lvl == "3":
                    info["l3"] = val
            info["source"] = "sysfs"
        elif system == "Darwin":
            def sc(k):
                o = _run(["sysctl", "-n", k])
                return int(o) if o.isdigit() else None
            info["line"] = sc("hw.cachelinesize")
            info["l1d"] = sc("hw.perflevel0.l1dcachesize") or sc("hw.l1dcachesize")
            info["l2"] = sc("hw.perflevel0.l2cachesize") or sc("hw.l2cachesize")
            info["l3"] = sc("hw.l3cachesize")
            info["source"] = "sysctl"
    except Exception as ex:
        info["source"] = f"detection failed: {ex}"
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


def fmt_bytes(x):
    if not x:
        return "n/a"
    if x < 1024:
        return f"{x} B"
    if x < 1024 ** 2:
        k = x / 1024
        return f"{k:.0f} KiB" if k == int(k) else f"{k:.1f} KiB"
    return f"{x / 1024 ** 2:.2f} MiB"


def pow2_sweep(plog: int):
    """{1, 2, 4, ..., P_log} (P_log appended if not a power of two)."""
    ps, p = [], 1
    while p < plog:
        ps.append(p)
        p *= 2
    ps.append(plog)
    return ps


def save_json(name: str, data) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_json(name: str):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def header(title: str):
    print("=" * 76)
    print(title)
    print(f"Student: {STUDENT_NAME}  ID: {STUDENT_ID}")
    print(f"Python {platform.python_version()} | {platform.system()} {platform.release()} | "
          f"start method: {mp.get_start_method()}")
    print("=" * 76)


def common_args(parser, warmup=3, runs=9):
    parser.add_argument("--warmup", type=int, default=warmup)
    parser.add_argument("--runs", type=int, default=runs)
    parser.add_argument("--pphys", type=int, default=None, help="override physical core count")
    return parser


def resolve_pphys(args) -> int:
    return args.pphys if args.pphys else get_physical_cores()
