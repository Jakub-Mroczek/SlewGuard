import ctypes
import numpy as np
import os
import time
import threading
import pynvml

# --- Load library ---
lib_path = os.path.join(os.path.dirname(__file__), "lib", "libgemm.so")
lib = ctypes.CDLL(lib_path)

lib.run_gemm.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
lib.run_gemm.restype = None

# --- NVML setup ---
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def poll_gpu(readings, stop_event, interval=0.01):
    """Poll GPU telemetry until stop_event is set."""
    while not stop_event.is_set():
        t = time.time()
        power_w    = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        temp_c     = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        util_pct   = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        clock_mhz  = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        readings.append((t, power_w, temp_c, util_pct, clock_mhz))
        time.sleep(interval)

def run_and_profile(A, B, batch_size, repeat=50, poll_interval=0.01):
    """
    Run GEMM kernel `repeat` times with given batch_size while polling GPU.
    Returns list of (timestamp, power_w, temp_c, util_pct, clock_mhz).
    """
    M = batch_size
    K, N = B.shape

    C = np.zeros((M, N), dtype=np.float32)
    A_batch = np.ascontiguousarray(A[:M])

    a_ptr = A_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_ptr = C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    readings = []
    stop_event = threading.Event()

    # Start polling thread
    poller = threading.Thread(target=poll_gpu, args=(readings, stop_event, poll_interval))
    poller.start()

    # Run kernel repeatedly so poller has time to sample
    for _ in range(repeat):
        lib.run_gemm(a_ptr, b_ptr, c_ptr,
                     ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K))

    stop_event.set()
    poller.join()

    return readings


if __name__ == "__main__":
    N = 1024
    A = np.ones((N, N), dtype=np.float32)
    B = np.ones((N, N), dtype=np.float32)

    batch_sizes = [64, 128, 256, 512, 1024]

    print(f"{'batch_size':>12} {'samples':>8} {'avg_power_W':>12} {'max_power_W':>12} {'avg_util%':>10} {'avg_clock_MHz':>14}")
    print("-" * 70)

    for batch_size in batch_sizes:
        readings = run_and_profile(A, B, batch_size, repeat=50)

        powers  = [r[1] for r in readings]
        utils   = [r[3] for r in readings]
        clocks  = [r[4] for r in readings]

        print(f"{batch_size:>12} {len(readings):>8} {np.mean(powers):>12.1f} {np.max(powers):>12.1f} {np.mean(utils):>10.1f} {np.mean(clocks):>14.1f}")

    pynvml.nvmlShutdown()
