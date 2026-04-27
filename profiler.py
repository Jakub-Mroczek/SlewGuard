import argparse
import ctypes
import numpy as np
import os
import time
import threading
import pynvml


# --- Load library ---
lib_path = os.path.join(os.path.dirname(__file__), "lib", "libgemm.so")
lib = ctypes.CDLL(lib_path)

# --- Kernel registry API ---
lib.kernel_count.argtypes = []
lib.kernel_count.restype  = ctypes.c_int

lib.kernel_name.argtypes = [ctypes.c_int]
lib.kernel_name.restype  = ctypes.c_char_p

lib.kernel_active.argtypes = []
lib.kernel_active.restype  = ctypes.c_int

lib.kernel_select_by_index.argtypes = [ctypes.c_int]
lib.kernel_select_by_index.restype  = ctypes.c_int

lib.kernel_select_by_name.argtypes = [ctypes.c_char_p]
lib.kernel_select_by_name.restype  = ctypes.c_int

# --- Persistent-buffer API (dispatches to the active kernel) ---
lib.gemm_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.gemm_init.restype  = ctypes.c_int

lib.gemm_upload_A.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.gemm_upload_A.restype  = ctypes.c_int

lib.gemm_upload_B.argtypes = [ctypes.POINTER(ctypes.c_float)]
lib.gemm_upload_B.restype  = ctypes.c_int

lib.gemm_run.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
lib.gemm_run.restype  = ctypes.c_int

lib.gemm_destroy.argtypes = []
lib.gemm_destroy.restype  = ctypes.c_int


# --- Registry helpers ---
def list_kernels():
    """Return [(idx, name), ...] for all registered kernels."""
    out = []
    for i in range(lib.kernel_count()):
        name = lib.kernel_name(i)
        out.append((i, name.decode("utf-8") if name else f"<idx{i}>"))
    return out


def select_kernel(name: str) -> int:
    """Select the active kernel by name. Returns the chosen index."""
    rc = lib.kernel_select_by_name(name.encode("utf-8"))
    if rc < 0:
        names = ", ".join(n for _, n in list_kernels())
        raise ValueError(f"Unknown kernel '{name}'. Available: {names}")
    return rc


# --- NVML setup ---
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)


def poll_gpu(readings, stop_event, interval=0.01):
    """Poll GPU telemetry until stop_event is set."""
    while not stop_event.is_set():
        t = time.time()
        power_w   = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        temp_c    = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        util_pct  = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        readings.append((t, power_w, temp_c, util_pct, clock_mhz))
        time.sleep(interval)


# NVML internal power-sample ring buffer — the driver records power telemetry at the GPU's native rate (typically 50-200 Hz) with no cache duplicates. 

_POWER_SAMPLE_TYPE = next(
    (getattr(pynvml, n) for n in (
        "NVML_TOTAL_POWER_SAMPLES",
        "NVML_GPU_POWER_SAMPLES",
    ) if hasattr(pynvml, n)),
    None,
)


def fetch_nvml_power_samples(last_ts_us: int = 0):
    """Drain the NVML power ring buffer since last_ts_us.

    Returns (samples, newest_ts_us):
        samples      = [(timestamp_us, power_w), ...] with timestamp strictly greater than last_ts_us, chronological.
        newest_ts_us = max timestamp seen (pass back on the next call).
    """
    if _POWER_SAMPLE_TYPE is None:
        return [], last_ts_us
    try:
        _, samples = pynvml.nvmlDeviceGetSamples(
            handle, _POWER_SAMPLE_TYPE, last_ts_us,
        )
    except pynvml.NVMLError:
        return [], last_ts_us
    out = []
    newest = last_ts_us
    for s in samples:
        ts = int(s.timeStamp)
        sv = s.sampleValue
        # Power samples are mW; pynvml union field is uiVal or ulVal.
        mw = getattr(sv, "uiVal", None)
        if mw is None:
            mw = getattr(sv, "ulVal", None)
        if mw is None:
            continue
        out.append((ts, float(mw) / 1000.0))
        if ts > newest:
            newest = ts
    return out, newest


def run_and_profile(A, batch_size, repeat=50, poll_interval=0.01):
    """Launch the active kernel `repeat` times at the requested batch size on
    persistent device buffers. Per-launch GPU time is measured via CUDA events.

    Returns (readings, kernel_ms).
    """
    M = batch_size
    A_batch = np.ascontiguousarray(A[:M])
    a_ptr = A_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    if lib.gemm_upload_A(a_ptr, M) != 0:
        raise RuntimeError("gemm_upload_A failed")

    readings = []
    stop_event = threading.Event()
    poller = threading.Thread(target=poll_gpu,
                              args=(readings, stop_event, poll_interval))
    poller.start()

    kernel_ms = np.empty(repeat, dtype=np.float32)
    ms = ctypes.c_float(0.0)
    try:
        for i in range(repeat):
            rc = lib.gemm_run(ctypes.c_int(M), ctypes.byref(ms))
            if rc != 0:
                raise RuntimeError(f"gemm_run failed (rc={rc}) at iter {i}")
            kernel_ms[i] = ms.value
    finally:
        stop_event.set()
        poller.join()

    return readings, kernel_ms


# --- Fixed sweep parameters ---
_N   = 1024
_BATCH_SIZES = [64, 128, 256, 512, 1024]
_REPEAT = 50
_POLL_INTERVAL = 0.01


def _argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SlewGuard kernel sweep profiler")
    p.add_argument("--kernel", type=str, default="naive_gemm",
                   help="Registered kernel name to run.")
    p.add_argument("--list-kernels", action="store_true",
                   help="List registered kernels and exit.")
    return p


if __name__ == "__main__":
    args = _argparser().parse_args()

    if args.list_kernels:
        for idx, name in list_kernels():
            print(f"  [{idx}] {name}")
        pynvml.nvmlShutdown()
        raise SystemExit(0)

    select_kernel(args.kernel)
    print(f"Active kernel: {args.kernel}")

    A = np.ones((_N, _N), dtype=np.float32)
    B = np.ones((_N, _N), dtype=np.float32)

    if lib.gemm_init(_N, _N, _N) != 0:
        raise RuntimeError("gemm_init failed")

    try:
        b_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        if lib.gemm_upload_B(b_ptr) != 0:
            raise RuntimeError("gemm_upload_B failed")

        header = (f"{'batch_size':>12} {'samples':>8} {'avg_power_W':>12} "
                  f"{'max_power_W':>12} {'avg_util%':>10} {'avg_clock_MHz':>14} "
                  f"{'avg_kern_ms':>12} {'GFLOP/s':>10}")
        print(header)
        print("-" * len(header))

        for batch_size in _BATCH_SIZES:
            readings, kernel_ms = run_and_profile(
                A, batch_size, repeat=_REPEAT, poll_interval=_POLL_INTERVAL
            )
            powers = [r[1] for r in readings]
            utils  = [r[3] for r in readings]
            clocks = [r[4] for r in readings]
            avg_ms = float(np.mean(kernel_ms))
            gflops = (2.0 * batch_size * _N * _N) / (avg_ms * 1e6) if avg_ms > 0 else 0.0
            print(f"{batch_size:>12} {len(readings):>8} {np.mean(powers):>12.1f} "
                  f"{np.max(powers):>12.1f} {np.mean(utils):>10.1f} "
                  f"{np.mean(clocks):>14.1f} {avg_ms:>12.3f} {gflops:>10.1f}")
    finally:
        lib.gemm_destroy()
        pynvml.nvmlShutdown()
