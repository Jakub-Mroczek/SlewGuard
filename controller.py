"""SlewGuard closed-loop controller

Drives a registered GEMM kernel continuously and throttles the batch size M with a proactive hysteresis controller so V_load stays above V_min.

Threads:
kernel worker — continuously launches the active GEMM at the latest updated M (this is the workload).
NVML poller  — samples real GPU power.
controller   — main thread; one PDN step + one decision per tick.

Controller: 
Two-phase run:
Phase 1 : open-loop linear ramp M_MIN -> M_MAX over --ramp-up-s
Phase 2 : closed-loop PDN-feedback throttling until --total-seconds

Controller trigger happens at V_shrink = V_safe − deadband
Deadband is there to ensure batch size does not throttle when v_load is hovering around V_safe. This can be tuned based on workload.

"""

from __future__ import annotations

import argparse
import csv
import ctypes
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from pdn import PDN
from profiler import (
    lib, poll_gpu, select_kernel, list_kernels,
    fetch_nvml_power_samples,
)



# Fixed run parameters

_N = 512
_M_MIN = 64
_M_MAX = 512
_M_INIT = _M_MIN

# Open-loop warm-up ramp (M_MIN -> M_MAX) before closed-loop takes over.
_DEFAULT_RAMP_UP_S = 1.0

# Shrink step is a ceiling (not auto-sized).  
# Grow step is auto-sized in run_live() from the measured PDN envelope so a single grow's V_ac overshoot stays inside 0.5·min(V_load(M_MIN)−V_safe, V_safe−V_min).
_SHRINK_STEP_DEFAULT = 64
_GROW_STEP_DEFAULT = 128

# V_safe = V_min + _V_SAFE_MARGIN_MV (default 20 mV), hard-capped at +50 mV.
# Clamped at runtime to stay:
# V_min + 3 mV  <=  V_safe  <=  V_load(M_MIN) − 3 mV
_V_SAFE_MARGIN_MV = 30.0
_V_SAFE_MARGIN_MV_CAP = 60.0

# Deadband below V_safe (V_shrink = V_safe − _V_SAFE_HYSTERESIS_MV).
# Shrinks fire below V_shrink; grows fire above it — so growing is ACTIVE
# inside the band and M keeps stepping
# Runtime-clamped so V_shrink => V_min + 3 mV.
_V_SAFE_HYSTERESIS_MV = 15.0

# PDN envelope time constant
# (f_LC ≈ 41 kHz at L=15 nH, C=1 mF -> ~200 us settling).
_TAU_ENV_S = 200e-6

# NVML calibration windowing (_calibrate_pdn_from_nvml)
_CALIB_WARMUP_S = 0.10
_CALIB_SAMPLE_S = 0.25
_CALIB_POLL_S = 0.01

_OUT_PREFIX = "slewguard_run"



# PDN-model auto-calibration from real NVML telemetry.

def _phase_stats(
    samples: list[tuple[float, float]],
) -> tuple[Optional[float], Optional[float], int]:
    """Return (mean_w, sigma_w, n_used) for (t_s, power_w) samples.
    """
    if not samples:
        return None, None, 0
    uniq: list[tuple[float, float]] = []
    last_p = None
    for t, p in samples:
        if p != last_p:
            uniq.append((t, p))
            last_p = p
    if len(uniq) < 2:
        mean = float(uniq[0][1]) if uniq else None
        return mean, 0.0, len(uniq)
    t_arr = np.asarray([s[0] for s in uniq], dtype=np.float64)
    p_arr = np.asarray([s[1] for s in uniq], dtype=np.float64)
    slope, intercept = np.polyfit(t_arr, p_arr, 1)
    residuals = p_arr - (slope * t_arr + intercept)
    return float(p_arr.mean()), float(np.std(residuals, ddof=1)), len(uniq)


def _calibrate_pdn_from_nvml(
    run_fn,
    M_min: int,
    M_max: int,
    warmup_s: float = _CALIB_WARMUP_S,
    sample_s: float = _CALIB_SAMPLE_S,
    poll_s: float = _CALIB_POLL_S,
) -> dict:
    # Measure P_idle, P_full and σ(M_min), σ(M_max) via NVML.

    readings: list = []
    stop_ev = threading.Event()
    poller = threading.Thread(
        target=poll_gpu, args=(readings, stop_ev, poll_s), daemon=True,
    )
    poller.start()

    phase_samples: dict[int, list[tuple[float, float]]] = {M_min: [], M_max: []}
    used_buffer = True

    def _run_phase(M: int) -> tuple[float, float]:
        nonlocal used_buffer
        _, last_ts = fetch_nvml_power_samples(0)
        t0 = time.time()
        while time.time() < t0 + warmup_s:
            if run_fn(M)[0] != 0:
                break
        _, last_ts = fetch_nvml_power_samples(last_ts)
        t_smp_s = time.time()
        t_end   = t_smp_s + sample_s
        while time.time() < t_end:
            if run_fn(M)[0] != 0:
                break
        buf_samples, _ = fetch_nvml_power_samples(last_ts)
        if buf_samples:
            for ts_us, pw in buf_samples:
                phase_samples[M].append((ts_us * 1e-6, pw))
        else:
            used_buffer = False
            for r in readings:
                if t_smp_s <= r[0] <= t_end:
                    phase_samples[M].append((r[0], r[1]))
        return t_smp_s, t_end

    try:
        _run_phase(M_min)
        _run_phase(M_max)
    finally:
        stop_ev.set()
        poller.join()

    P_idle, P_idle_std, n_idle = _phase_stats(phase_samples[M_min])
    P_full, P_full_std, n_full = _phase_stats(phase_samples[M_max])
    return {
        "P_idle": P_idle,
        "P_full": P_full,
        "P_idle_std": P_idle_std,
        "P_full_std": P_full_std,
        "source": "nvml_buffer" if used_buffer else "poll_fallback",
        "n_samples_idle": n_idle,
        "n_samples_full": n_full,
    }


def _validate_calibration(
    calib: dict,
) -> tuple[float, float, float, float]:
    """Validate and return (P_idle, P_full, P_idle_std, P_full_std).

    Raises RuntimeError on missing/implausible means (P_idle > 1 W,
    P_full > P_idle + 3 W, P_full < 2000 W).  Stdevs are reported as-is.
    """
    pi = calib.get("P_idle")
    pf = calib.get("P_full")
    pi_s = calib.get("P_idle_std") or 0.0
    pf_s = calib.get("P_full_std") or 0.0

    def _bad(msg: str) -> None:
        raise RuntimeError(
            f"NVML PDN calibration failed: {msg}.  "
            f"Measured P_idle={pi}, P_full={pf}.  "
            f"Ensure NVML power telemetry is enabled and the kernel is "
            f"drawing measurable load."
        )

    if pi is None or pf is None:
        _bad("NVML returned no samples for at least one calibration phase")
    if pi <= 1.0:
        _bad(f"P_idle={pi:.2f} W is implausibly low (is the GEMM actually running?)")
    if pf <= pi + 3.0:
        _bad(f"P_full={pf:.2f} W is not measurably above P_idle={pi:.2f} W "
             "— kernel is not scaling with batch size, NVML may be clamped")
    if pf >= 2000.0:
        _bad(f"P_full={pf:.2f} W is outside the plausible GPU power range")
    return float(pi), float(pf), float(pi_s), float(pf_s)


# tick  = 50 us — realistic software closed-loop, 4× faster than tau_env
# shrink_period  = 1 ms — 5·tau_env: one LC ring -> one shrink, realistic latency
# grow_period = 200 us = tau_env: each grow's V_ac decays before the next,
# so V_ac bursts cannot accumulate past V_safe−V_min.

_DEFAULT_TICK_US  = 50.0
_DEFAULT_POLL_US   = 500.0
_DEFAULT_SHRINK_PERIOD_US = 5 * _TAU_ENV_S * 1e6 # 1000 us
_DEFAULT_GROW_PERIOD_US = _TAU_ENV_S * 1e6  #  200 us (= tau_env)
_DEFAULT_PDN_NOISE_SCALE = 1.0


# 2-state hysteresis controller  (single V_safe trigger)

@dataclass
class HysteresisController:
	
    """2-state hysteresis throttle with a single operational threshold.
    V_shrink = V_safe − deadband is the only trigger:
    """
    
    V_safe: float
    shrink_period_s: float
    grow_period_s: float
    V_deadband_v: float = _V_SAFE_HYSTERESIS_MV * 1e-3
    grow_step: int   = _GROW_STEP_DEFAULT
    shrink_step: int   = _SHRINK_STEP_DEFAULT
    M: int = _M_INIT
    M_min: int = _M_MIN
    M_max: int = _M_MAX
    throttle_enabled: bool  = True
    _last_shrink_t: float = field(default=-1.0, repr=False)
    _last_grow_t: float = field(default=-1.0, repr=False)

    @property
    def V_shrink_trigger(self) -> float:
        return self.V_safe - self.V_deadband_v

    def update(self, V_load: float, t_now: float) -> tuple[int, str]:
        if not self.throttle_enabled:
            return self.M, "hold"

        if V_load < self.V_shrink_trigger:
            if (t_now - self._last_shrink_t) >= self.shrink_period_s:
                self._last_shrink_t = t_now
                self.M = max(self.M_min, self.M - self.shrink_step)
                self._last_grow_t = t_now + self.shrink_period_s
                return self.M, "shrink"
            return self.M, "hold"

        if self.M < self.M_max and (t_now - self._last_grow_t) >= self.grow_period_s:
            self._last_grow_t = t_now
            self.M = min(self.M_max, self.M + self.grow_step)
            return self.M, "grow"
        return self.M, "hold"



# Telemetry containers

@dataclass
class Tick:
    t:  float   # seconds since run start
    M:  int     # commanded batch size at end of window
    power_w: float   # model load power at end of window [W]
    V_load:  float   # worst PDN V_load this window [V]
    V_load_avg: float   # mean  PDN V_load this window [V]


@dataclass
class Event:
    t:  float
    kind:  str    # "shrink" | "grow"
    from_M: int
    to_M:  int
    t_violation: Optional[float]
    latency_s: float = 0.0



# Kernel worker — continuous GEMM at the latest M

@dataclass
class WorkerState:
    M_current:       int   = _M_INIT
    total_flops:     float = 0.0
    total_kernel_ms: float = 0.0
    n_launches:      int   = 0
    lock: threading.Lock   = field(default_factory=threading.Lock)


def _kernel_worker(
    state: WorkerState,
    stop: threading.Event,
    run_fn,
    flops_fn,
    M_min: int,
) -> None:
    """Continuously launch the workload at state.M_current.  The GPU sync
    inside run_fn releases the GIL so the controller thread runs in parallel."""
    while not stop.is_set():
        M = max(M_min, state.M_current)
        rc, elapsed_ms = run_fn(M)
        if rc != 0:
            break
        with state.lock:
            state.total_flops     += flops_fn(M)
            state.total_kernel_ms += elapsed_ms
            state.n_launches      += 1



# Precise wait

def _wait_until(deadline: float) -> None:
    while True:
        left = deadline - time.perf_counter()
        if left <= 0:
            return
        if left > 5e-4:
            time.sleep(left * 0.5)
        else:
            time.sleep(0)    # yield GIL but spin tight



# Live control loop

def run_live(
    R: float, L: float, C: float, V_min: float,
    total_seconds: float, kernel_name: str,
    tick_us: float             = _DEFAULT_TICK_US,
    poll_us: float             = _DEFAULT_POLL_US,
    grow_period_us: float      = _DEFAULT_GROW_PERIOD_US,
    shrink_period_us: float    = _DEFAULT_SHRINK_PERIOD_US,
    throttle_enabled: bool     = True,
    ramp_up_s: float           = _DEFAULT_RAMP_UP_S,
    deadband_mv: float         = _V_SAFE_HYSTERESIS_MV,
    pdn_noise_scale: float     = _DEFAULT_PDN_NOISE_SCALE,
    workload: str              = "gemm",
    M_min: int                 = _M_MIN,
    M_max: int                 = _M_MAX,
) -> tuple:
    """
    Run the closed-loop SlewGuard controller on the live GPU.

	Two-phase run:
	Phase 1 : open-loop linear ramp M_MIN -> M_MAX over --ramp-up-s
	Phase 2 : closed-loop PDN-feedback throttling until --total-seconds

	Controller trigger happens at V_shrink = V_safe − deadband
	Deadband is there to ensure batch size does not throttle when v_load is hovering around V_safe. 
	This can be tuned based on workload.

    """
    tick_s          = tick_us          * 1e-6
    poll_s          = poll_us          * 1e-6
    grow_period_s   = grow_period_us   * 1e-6
    shrink_period_s = shrink_period_us * 1e-6
    record_period_s = max(tick_s, 1e-3)
    ramp_up_s       = max(0.0, float(ramp_up_s))

    pdn = PDN(R=R, L=L, C=C, V_min=V_min, _TAU_ENV=_TAU_ENV_S)

    timeline:        List[Tick]  = []
    events:          List[Event] = []
    nvml_rdg:        list        = []
    stop_ev          = threading.Event()
    poller = kernel_thr = None

    # Build workload backend: run_fn(M) -> (rc, elapsed_ms), flops_fn(M) -> float
    _resnet_wl = None
    if workload == "gemm":
        select_kernel(kernel_name)
        A = np.ones((_N, _N), dtype=np.float32)
        B = np.ones((_N, _N), dtype=np.float32)
        if lib.gemm_init(_N, _N, _N) != 0:
            raise RuntimeError("gemm_init failed")
        _ms_buf = ctypes.c_float(0.0)
        def run_fn(M: int):
            rc = lib.gemm_run(ctypes.c_int(M), ctypes.byref(_ms_buf))
            return rc, float(_ms_buf.value)
        def flops_fn(M: int) -> float:
            return 2.0 * M * _N * _N
        workload_label = kernel_name
    elif workload == "resnet":
        from resnet_workload import ResNetWorkload
        _resnet_wl = ResNetWorkload(M_max=M_max)
        run_fn = _resnet_wl.run
        flops_fn = _resnet_wl.flops
        workload_label = f"resnet18 (M_min={M_min}, M_max={M_max})"
    else:
        raise ValueError(f"Unknown workload '{workload}'. Choose 'gemm' or 'resnet'.")

    def _destroy_workload():
        if workload == "gemm":
            lib.gemm_destroy()
        elif _resnet_wl is not None:
            _resnet_wl.destroy()

    try:
        if workload == "gemm":
            b_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            a_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            if lib.gemm_upload_B(b_ptr) != 0:
                raise RuntimeError("gemm_upload_B failed")
            if lib.gemm_upload_A(a_ptr, _N) != 0:
                raise RuntimeError("gemm_upload_A failed")

        # Calibrate the PDN power model from the workload itself
        # (runs workload at M_min then M_max under NVML telemetry).
        calib_total = 2 * (_CALIB_WARMUP_S + _CALIB_SAMPLE_S)
        print(f"Calibrating PDN from workload "
              f"({calib_total:.1f} s NVML, {workload_label})...")
        calib = _calibrate_pdn_from_nvml(run_fn, M_min, M_max)
        P_idle, P_full, P_idle_std, P_full_std = _validate_calibration(calib)
        print(f"Calibration: source={calib.get('source', 'unknown')}  "
              f"n_samples idle/full = {calib.get('n_samples_idle', 0)}/"
              f"{calib.get('n_samples_full', 0)}  "
              f"P_idle={P_idle:.1f}±{P_idle_std:.2f}W  "
              f"P_full={P_full:.1f}±{P_full_std:.2f}W")

        # Closures over the measured envelope: deterministic DC level + NVML jitter.
        # Both are linearly interpolated between M_min and M_max endpoints.
        def load_power(M: int) -> float:
            if M_max == M_min:
                return P_idle
            frac = max(0.0, min(1.0, (M - M_min) / (M_max - M_min)))
            return P_idle + frac * (P_full - P_idle)

        def noise_sigma(M: int) -> float:
            if M_max == M_min:
                return P_idle_std
            frac = max(0.0, min(1.0, (M - M_min) / (M_max - M_min)))
            return P_idle_std + frac * (P_full_std - P_idle_std)

        # Auto-calibrate V_safe and GROW_STEP from the NVML envelope

        V_nom         = pdn.V_nom
        V_load_max_dc = V_nom - P_idle * R
        V_load_min_dc = V_nom - P_full * R

        # V_safe = V_min + margin (capped at 50 mV), clamped to stay above
        # V_min + 3 mV (safety) and below V_load(M_min) − 3 mV (achievable).
        margin_mv = min(_V_SAFE_MARGIN_MV, _V_SAFE_MARGIN_MV_CAP)
        V_safe_desired = V_min + margin_mv * 1e-3
        V_safe_floor   = V_min + 0.003
        V_safe_ceiling = min(V_load_max_dc - 0.003, V_min + _V_SAFE_MARGIN_MV_CAP * 1e-3)
        if V_safe_ceiling <= V_safe_floor:
            V_safe = max(V_safe_floor, V_load_max_dc - 0.003)
            print(f"WARNING: DEGENERATE PDN — V_load(M_min)={V_load_max_dc:.3f} V "
                  f"is at or below V_min+3 mV ({V_min+0.003:.3f} V).  V_min "
                  f"violations are physically UNAVOIDABLE for this "
                  f"(R, P_idle, V_min) triple at any batch size.  V_safe "
                  f"pinned to {V_safe:.3f} V; controller will drive M -> M_min "
                  f"to minimise the violation rate.  Reduce R, reduce kernel "
                  f"P_idle, or raise V_min to restore a valid operating band.")
        else:
            V_safe = max(V_safe_floor, min(V_safe_desired, V_safe_ceiling))

        # Auto-size the grow step so one grow's V_ac overshoot can't dip
        # V_load below V_safe or V_min:
        #   budget = 0.5 · min(V_load(M_min) − V_safe, V_safe − V_min)
        headroom_above_vsafe = max(V_load_max_dc - V_safe, 0.001)
        headroom_vsafe_vmin  = max(V_safe - V_min, 0.001)
        vac_budget           = 0.5 * min(headroom_above_vsafe, headroom_vsafe_vmin)
        dM_span          = max(1, M_max - M_min)
        slope_v_per_step = (P_full - P_idle) / dM_span * pdn.Z_LC * pdn.overshoot
        if slope_v_per_step > 0:
            grow_step_auto = int(vac_budget / slope_v_per_step)
        else:
            grow_step_auto = _GROW_STEP_DEFAULT
        grow_step = max(1, min(_GROW_STEP_DEFAULT, grow_step_auto))

        # Shrink step: capped to a quarter of the M range so it's sensible
        # for both wide (GEMM) and narrow (ResNet) M spans.
        shrink_step = min(_SHRINK_STEP_DEFAULT, max(1, dM_span // 4))

        # Effective deadband — clamped so V_shrink stays ≥ V_min + 3 mV.
        requested_db_v = max(0.0, deadband_mv) * 1e-3
        max_db_v       = max(0.0, (V_safe - V_min) - 0.003)
        V_deadband_v   = min(requested_db_v, max_db_v)
        if V_deadband_v < requested_db_v - 1e-9:
            print(f"  deadband clamp  : requested {deadband_mv:.1f} mV "
                  f"reduced to {V_deadband_v*1e3:.1f} mV so V_shrink stays "
                  f"≥ V_min+3 mV (V_safe − V_min is only "
                  f"{(V_safe - V_min)*1e3:.1f} mV — raise --V-min or lower "
                  f"--deadband-mv to restore the full deadband)")

        ctrl = HysteresisController(
            V_safe=V_safe,
            shrink_period_s=shrink_period_s,
            grow_period_s=grow_period_s,
            V_deadband_v=V_deadband_v,
            grow_step=grow_step,
            shrink_step=shrink_step,
            M_min=M_min,
            M_max=M_max,
            throttle_enabled=throttle_enabled,
        )
        wstate = WorkerState(M_current=M_min)

        # Natural di/dt from one grow step (the controller's own action on
        # the kernel's commanded M is the primary transient source).
        dP_grow  = (grow_step / dM_span) * (P_full - P_idle)
        Vac_grow = dP_grow * pdn.Z_LC * pdn.overshoot

        # Preamble — shows the kernel-measured envelope + derived thresholds.
        mode_banner = ("THROTTLE ENABLED" if throttle_enabled
                       else "UNTHROTTLED BASELINE (controller disabled)")
        print(f"Active workload   : {workload_label}")
        print(f"Mode              : {mode_banner}")
        print(f"Data flow         : kernel -> NVML(power) -> PDN(R,L,C) -> V_load -> controller -> M")
        print(f"                  : (every PDN sample and throttle decision is driven by "
              f"NVML-measured kernel power)")
        print(f"Kernel-measured P : P(M_MIN)={P_idle:.1f} W  "
              f"P(M_MAX)={P_full:.1f} W  "
              f"ΔP_full={P_full - P_idle:.1f} W")
        _eff_sigma_idle = P_idle_std * pdn_noise_scale
        _eff_sigma_full = P_full_std * pdn_noise_scale
        if pdn_noise_scale > 0.0:
            print(f"NVML jitter σ     : σ(M_MIN)={P_idle_std:.2f} W  "
                  f"σ(M_MAX)={P_full_std:.2f} W  "
                  f"(×scale={pdn_noise_scale:.2f} -> eff "
                  f"[{_eff_sigma_idle:.2f},{_eff_sigma_full:.2f}] W  "
                  f"V_load jitter ≈ σ·Z_LC·overshoot = "
                  f"[{_eff_sigma_idle*pdn.Z_LC*pdn.overshoot*1e3:.2f},"
                  f"{_eff_sigma_full*pdn.Z_LC*pdn.overshoot*1e3:.2f}] mV)")
        else:
            print(f"NVML jitter σ     : DISABLED (--pdn-noise-scale=0) — "
                  f"p_drive is deterministic f(M); V_load WILL flat-line "
                  f"when M holds steady.  Pass --pdn-noise-scale 1 to "
                  f"inject the NVML-measured workload jitter "
                  f"(σ(M_MIN)={P_idle_std:.2f} W, σ(M_MAX)={P_full_std:.2f} W).")
        print(f"PDN               : R={R*1e3:.2f} mΩ  L={L*1e9:.1f} nH  "
              f"C={C*1e3:.3f} mF  tau_env={_TAU_ENV_S*1e6:.0f} us")
        print(f"                  : Z_LC={pdn.Z_LC*1e3:.2f} mΩ  "
              f"zeta={pdn.zeta:.3f}  overshoot={pdn.overshoot:.3f}")
        print(f"Reachable V_load  : [{V_load_min_dc:.3f}, {V_load_max_dc:.3f}] V  "
              f"(DC, M=M_MAX..M_MIN)")
        V_shrink_trigger = ctrl.V_shrink_trigger
        effective_db_mv  = ctrl.V_deadband_v * 1e3
        print(f"Thresholds        : V_min={V_min:.3f} V (violation ref)  "
              f"V_safe={V_safe:.3f} V (headroom ref)  "
              f"V_shrink={V_shrink_trigger:.3f} V (single operational trigger)")
        print(f"  deadband        : {effective_db_mv:.1f} mV gap V_safe->V_shrink; "
              f"shrink<V_shrink, grow≥V_shrink "
              f"(keeps di/dt alive inside band -> V_load keeps oscillating)")
        print(f"  V_safe headroom : {(V_safe - V_min)*1e3:.1f} mV above V_min  "
              f"({(V_load_max_dc - V_safe)*1e3:.1f} mV below V_load(M_MIN))")
        if V_load_min_dc < V_min:
            print(f"  V_load(M_MAX)   : {V_load_min_dc:.3f} V < V_min — "
                  f"kernel at M_MAX naturally violates V_min "
                  f"(demo: controller will throttle down)")
        elif V_load_min_dc > V_safe:
            print(f"  V_load(M_MAX)   : {V_load_min_dc:.3f} V > V_safe "
                  f"({V_safe:.3f} V) — PDN is healthier than V_min+"
                  f"{_V_SAFE_MARGIN_MV_CAP:.0f} mV; V_load never dips below "
                  f"V_safe so controller will stay in GROW-only mode and M -> "
                  f"M_MAX.  Raise --V-min or use a heavier kernel / larger "
                  f"--R to exercise throttling.")
        else:
            print(f"  V_load(M_MAX)   : {V_load_min_dc:.3f} V ≥ V_min — "
                  f"kernel does not violate V_min; "
                  f"throttling driven by grow-step overshoot only")
        print(f"Controller        : tick={tick_us:.1f} us  "
              f"grow_period={grow_period_us:.1f} us  "
              f"shrink_period={shrink_period_us:.0f} us")
        print(f"  step sizes      : shrink=-{shrink_step}  "
              f"grow=+{grow_step}  "
              f"(V_ac/grow ≤ 0.5·min(head_above_V_safe, V_safe−V_min)="
              f"{min(headroom_above_vsafe, headroom_vsafe_vmin)*1e3:.1f} mV)")
        print(f"  V_safe placement: V_min + {margin_mv:.1f} mV "
              f"(fixed offset, cap {_V_SAFE_MARGIN_MV_CAP:.0f} mV)")
        print(f"Natural di/dt     : grow step ΔP={dP_grow:.2f} W -> "
              f"V_ac={Vac_grow*1e3:.1f} mV overshoot per grow")
        if ramp_up_s > 0:
            print(f"Phase 1 (ramp)    : 0.0–{ramp_up_s:.2f} s  "
                  f"open-loop M: {M_min}->{M_max} "
                  f"({(M_max-M_min)/ramp_up_s:.0f} M/s)  "
                  f"— controller observes only")
            print(f"Phase 2 (control) : {ramp_up_s:.2f}–{total_seconds:.2f} s  "
                  f"closed-loop PDN-feedback throttling")
        else:
            print(f"Phase 2 (control) : 0.0–{total_seconds:.2f} s  "
                  f"closed-loop PDN-feedback throttling (no warm-up ramp)")

        # Start workers and enter the main control loop.
        t0    = time.perf_counter()
        wall0 = time.time()

        poller = threading.Thread(
            target=poll_gpu, args=(nvml_rdg, stop_ev, poll_s), daemon=True)
        poller.start()

        kernel_thr = threading.Thread(
            target=_kernel_worker,
            args=(wstate, stop_ev, run_fn, flops_fn, M_min),
            daemon=True)
        kernel_thr.start()

        next_tick   = t0 + tick_s
        next_record = t0 + record_period_s
        n_iters     = 0

        win_vmin:  float = pdn.V_nom
        win_vsum: float = 0.0
        win_vn: int   = 0
        win_psum:   float = 0.0
        win_pn:  int  = 0

        t_first_viol: Optional[float] = None

        dM_ramp   = max(1, M_max - M_min)
        in_ramp_prev = (ramp_up_s > 0)

        # Dedicated PRNG (seed=0) for reproducible NVML-grounded jitter.
        rng = random.Random(0)

        while time.perf_counter() - t0 < total_seconds:
            t_rel   = time.perf_counter() - t0
            n_iters += 1

            # On the ramp->control boundary, drop any stale V_shrink
            # crossing from the ramp — post-ramp latency should be
            # measured from a FRESH controller-visible crossing.
            in_ramp = t_rel < ramp_up_s
            if in_ramp_prev and not in_ramp:
                t_first_viol = None
            in_ramp_prev = in_ramp

            # p_drive = deterministic load_power(M) + NVML-grounded jitter.
            p_dc    = load_power(ctrl.M)
            p_noise = (rng.gauss(0.0, noise_sigma(ctrl.M)) * pdn_noise_scale
                       if pdn_noise_scale > 0.0 else 0.0)
            p_drive = max(0.0, p_dc + p_noise)
            v       = pdn.step(p_drive, dt=tick_s)

            win_vmin   = min(win_vmin, v)
            win_vsum  += v
            win_vn    += 1
            win_psum  += p_drive
            win_pn    += 1

            if v < ctrl.V_shrink_trigger and t_first_viol is None:
                t_first_viol = t_rel

            prev_M = ctrl.M

            # Phase 1: open-loop linear ramp (controller observes only).
            # Phase 2: closed-loop PDN-feedback control.
            if in_ramp:
                frac  = min(1.0, t_rel / ramp_up_s) if ramp_up_s > 0 else 1.0
                M_sched = M_min + int(round(dM_ramp * frac))
                M_sched = max(M_min, min(M_max, M_sched))
                if M_sched != ctrl.M:
                    ctrl.M = M_sched
                # Keep timers fresh so no action is pre-scheduled at ramp end.
                ctrl._last_shrink_t = t_rel
                ctrl._last_grow_t   = t_rel
                action = "ramp"
            else:
                _, action = ctrl.update(v, t_rel)

            if ctrl.M != prev_M:
                wstate.M_current = ctrl.M
                if ctrl.M < prev_M:
                    # Clamp same-tick latency to tick_s so we report "≤ 1 tick".
                    lat = max(t_rel - t_first_viol, tick_s) \
                          if t_first_viol is not None else tick_s
                    events.append(Event(
                        t=t_rel, kind=action,
                        from_M=prev_M, to_M=ctrl.M,
                        t_violation=t_first_viol, latency_s=lat,
                    ))
                    if action == "shrink":
                        t_first_viol = None
                elif action != "ramp":
                    events.append(Event(
                        t=t_rel, kind="grow",
                        from_M=prev_M, to_M=ctrl.M,
                        t_violation=None, latency_s=0.0,
                    ))

            if time.perf_counter() >= next_record:
                v_avg = win_vsum / win_vn if win_vn else pdn.V_nom
                p_avg = win_psum / win_pn if win_pn else 0.0
                timeline.append(Tick(
                    t=t_rel, M=ctrl.M,
                    power_w=p_avg, V_load=win_vmin, V_load_avg=v_avg,
                ))
                win_vmin     = pdn.V_nom
                win_vsum     = 0.0;  win_vn = 0
                win_psum     = 0.0;  win_pn = 0
                next_record += record_period_s

            next_tick += tick_s
            _wait_until(next_tick)

    finally:
        stop_ev.set()
        if poller:     poller.join(timeout=1.0)
        if kernel_thr: kernel_thr.join(timeout=2.0)
        _destroy_workload()

    with wstate.lock:
        total_flops = wstate.total_flops
        n_launches  = wstate.n_launches

    wall_s           = time.time() - wall0
    achieved_tick_us = (wall_s / n_iters * 1e6) if n_iters else float("inf")

    # Keep the NVML poll trace (already collected in the background by
    # poll_gpu) as (t_rel_s, power_w) pairs for the comparison plots.
    nvml_trace = [(r[0] - wall0, r[1]) for r in nvml_rdg]

    meta = {
        "V_nom":  pdn.V_nom,
        "V_min":  V_min,
        "V_safe":  V_safe,
        "V_shrink_trigger":  ctrl.V_shrink_trigger,
        "V_deadband_mv":  ctrl.V_deadband_v * 1e3,
        "kernel":  workload_label,
        "M_min":   M_min,
        "M_max":   M_max,
        "R":     R,  "L": L,  "C": C,
        "Z_LC":  pdn.Z_LC,
        "zeta":  pdn.zeta,
        "overshoot":   pdn.overshoot,
        "tick_us":  tick_us,
        "poll_us":  poll_us,
        "grow_period_us": grow_period_us,
        "shrink_period_us": shrink_period_us,
        "throttle_enabled":throttle_enabled,
        "P_idle_w":     P_idle,
        "P_full_w":   P_full,
        "P_idle_std_w":   P_idle_std,
        "P_full_std_w":   P_full_std,
        "pdn_noise_scale":  pdn_noise_scale,
        "dP_grow_w":     dP_grow,
        "Vac_grow_v":   Vac_grow,
        "shrink_step":   shrink_step,
        "grow_step":   grow_step,
        "ramp_up_s":   ramp_up_s,
        "V_load_max_dc":  V_load_max_dc,
        "V_load_min_dc":   V_load_min_dc,
        "total_flops":   total_flops,
        "total_seconds":   wall_s,
        "n_launches":   n_launches,
        "n_iters":  n_iters,
        "achieved_tick_us":  achieved_tick_us,
        "nvml_trace":  nvml_trace,
    }
    return timeline, events, meta



# CSV export

def write_csv(timeline: List[Tick], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "M", "power_w", "V_load_worst", "V_load_avg"])
        for tk in timeline:
            w.writerow([
                f"{tk.t:.6f}", tk.M, f"{tk.power_w:.3f}",
                f"{tk.V_load:.6f}", f"{tk.V_load_avg:.6f}",
            ])



# Plotting helpers

def _arrays(tl: List[Tick]):
    t  = np.array([x.t for x in tl])
    M  = np.array([x.M  for x in tl])
    P  = np.array([x.power_w for x in tl])
    Vw = np.array([x.V_load  for x in tl])
    Va = np.array([x.V_load_avg for x in tl])
    return t, M, P, Vw, Va


def _summarize(tl: List[Tick], ev: List[Event], meta: dict) -> dict:
    if not tl:
        return {}
    t  = np.array([x.t      for x in tl])
    M  = np.array([x.M      for x in tl])
    Vw = np.array([x.V_load for x in tl])

    total_s    = float(meta.get("total_seconds", tl[-1].t))
    avg_tflops = float(meta.get("total_flops", 0.0)) / max(total_s, 1e-9) / 1e12

    # Post-ramp slice so open-loop ramp doesn't skew controller metrics.
    ramp_up_s = float(meta.get("ramp_up_s", 0.0))
    ctrl_mask = t >= ramp_up_s

    shrinks = [e for e in ev if e.kind == "shrink"]
    grows   = [e for e in ev if e.kind == "grow"]
    lat_us  = [e.latency_s * 1e6 for e in shrinks if e.latency_s > 0]

    V_min_thr = meta["V_min"]
    viol_total = int((Vw < V_min_thr).sum())
    if ctrl_mask.any():
        viol_ctrl = int((Vw[ctrl_mask] < V_min_thr).sum())
        M_ctrl = M[ctrl_mask]
        M_avg  = float(M_ctrl.mean())
        M_min  = int(M_ctrl.min())
        M_max  = int(M_ctrl.max())
    else:
        viol_ctrl = viol_total
        M_avg = float(M.mean()); M_min = int(M.min()); M_max = int(M.max())

    return {
        "total_flops":  float(meta.get("total_flops", 0.0)),
        "total_s":  total_s,
        "avg_tflops":  avg_tflops,
        "n_shrinks":  len(shrinks),
        "n_grows":  len(grows),
        "n_violations":  viol_ctrl,        # post-ramp only
        "n_violations_total": viol_total,   # includes the ramp
        "avg_latency_us": float(np.mean(lat_us)) if lat_us else 0.0,
        "max_latency_us": float(np.max(lat_us))  if lat_us else 0.0,
        "M_avg":      M_avg,
        "M_min":  M_min,
        "M_max":   M_max,
        "ramp_up_s":  ramp_up_s,
    }


def plot_stacked(tl: List[Tick], ev: List[Event],
                 meta: dict, path: str, title_extra: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not tl:
        print("No data to plot.")
        return

    t, M, P, Vw, Va = _arrays(tl)
    s = _summarize(tl, ev, meta)

    V_nom  = meta["V_nom"]
    V_min  = meta["V_min"]
    V_safe = meta["V_safe"]

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 9))
    fig.subplots_adjust(hspace=0.06)

    # Shade the open-loop warm-up window on every panel.
    ramp_up_s = float(meta.get("ramp_up_s", 0.0))
    if ramp_up_s > 0:
        for a in ax:
            a.axvspan(0.0, ramp_up_s, color="tab:grey", alpha=0.10, zorder=0)

    # Panel 1: load model power.
    ax[0].plot(t, P, color="tab:red", linewidth=1.2,
               label=(f"load model power [W]  "
                      f"(P_idle={meta['P_idle_w']:.0f}±"
                      f"{meta.get('P_idle_std_w', 0.0):.1f} W, "
                      f"P_full={meta['P_full_w']:.0f}±"
                      f"{meta.get('P_full_std_w', 0.0):.1f} W, "
                      f"NVML-measured)"))
    if ramp_up_s > 0:
        ax[0].axvline(ramp_up_s, color="tab:grey", linestyle=":", linewidth=1.0)
        ax[0].text(
            ramp_up_s * 0.5, ax[0].get_ylim()[1],
            f"open-loop ramp\n0->{meta.get('M_max', _M_MAX)} in {ramp_up_s:.1f} s",
            ha="center", va="top", fontsize=8, color="dimgrey",
        )
    ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_ylabel("Load power [W]")
    ax[0].grid(True, alpha=0.3)

    # Panel 2: batch size M.
    ax[1].step(t, M, where="post", color="tab:blue", linewidth=1.6,
               label=f"M  (−{meta['shrink_step']}/shrink @ "
                     f"{meta['shrink_period_us']:.0f} us, "
                     f"+{meta['grow_step']}/grow @ "
                     f"{meta['grow_period_us']:.0f} us)")

    handles_M = [ax[1].get_lines()[0]]
    if ramp_up_s > 0:
        ax[1].axvline(ramp_up_s, color="tab:grey", linestyle=":", linewidth=1.0)
        handles_M.append(Line2D([0], [0], color="tab:grey", linestyle=":",
                                linewidth=1.0,
                                label=f"ramp->control @ {ramp_up_s:.1f} s"))
    ax[1].legend(handles=handles_M, loc="lower right", fontsize=8)
    ax[1].set_ylabel("Batch size M")
    ax[1].set_ylim(0, meta.get("M_max", _M_MAX) * 1.10)
    ax[1].grid(True, alpha=0.3)

    # Panel 3: V_load with V_safe / V_shrink / V_min reference lines.
    ax[2].plot(t, Vw, color="tab:green", linewidth=1.5,
               label="V_load worst-per-window")
    ax[2].plot(t, Va, color="tab:olive", linewidth=0.9, linestyle=":",
               label="V_load avg-per-window")

    ax[2].axhline(V_safe, color="tab:orange", linestyle="--", linewidth=1.4,
                  label=f"V_safe ({V_safe:.3f})  headroom ref")
    V_shrink = meta.get("V_shrink_trigger", V_safe)
    deadband_mv = meta.get("V_deadband_mv", 0.0)
    if V_shrink < V_safe - 1e-6:
        ax[2].axhline(V_shrink, color="tab:orange", linestyle=":",
                      linewidth=1.2,
                      label=(f"V_shrink ({V_shrink:.3f} = V_safe − "
                             f"{deadband_mv:.0f} mV)  shrink↓ / grow↑ trigger"))
    ax[2].axhline(V_min,  color="tab:red",    linestyle="--", linewidth=1.2,
                  label=f"V_min ({V_min:.3f})  safety reference")
    ax[2].fill_between(t, Vw, V_min, where=(Vw < V_min),
                       color="red", alpha=0.40, label="V_min violation")
    if ramp_up_s > 0:
        ax[2].axvline(ramp_up_s, color="tab:grey", linestyle=":", linewidth=1.0)

    # Y-axis zoom: show full reachable V_load band + V_safe / V_min lines.
    V_reach_lo = meta.get("V_load_min_dc", V_min)
    V_reach_hi = meta.get("V_load_max_dc", V_safe)
    v_lo = min(float(Vw.min()), V_min,  V_reach_lo, V_shrink) - 0.005
    v_hi = max(float(Va.max()), V_safe, V_reach_hi) + 0.005
    ax[2].set_ylim(v_lo, v_hi)
    ax[2].set_ylabel("V_load [V]")
    ax[2].set_xlabel("Time [s]")
    ax[2].grid(True, alpha=0.3)
    ax[2].legend(loc="lower right", fontsize=7, ncol=2)

    throttle_on = meta.get("throttle_enabled", True)
    mode_tag    = "SlewGuard (THROTTLE ON)" if throttle_on \
                  else "UNTHROTTLED BASELINE — controller disabled"
    title = (
        f"{mode_tag}  —  kernel={meta['kernel']}  —  "
        f"R={meta['R']*1e3:.2f} mΩ, L={meta['L']*1e9:.1f} nH, "
        f"C={meta['C']*1e3:.3f} mF  |  "
        f"Z_LC={meta['Z_LC']*1e3:.2f} mΩ, zeta={meta['zeta']:.3f}, "
        f"overshoot={meta['overshoot']:.2f}\n"
        f"tick={meta['tick_us']:.1f} us (achieved {meta['achieved_tick_us']:.1f} us)  |  "
        f"NVML-measured: P_idle={meta['P_idle_w']:.0f}±"
        f"{meta.get('P_idle_std_w', 0.0):.1f} W, "
        f"P_full={meta['P_full_w']:.0f}±"
        f"{meta.get('P_full_std_w', 0.0):.1f} W  "
        f"(jitter scale ×{meta.get('pdn_noise_scale', 0.0):.1f})  |  "
        f"reachable V_load "
        f"[{meta.get('V_load_min_dc', float('nan')):.3f}, "
        f"{meta.get('V_load_max_dc', float('nan')):.3f}] V  |  "
        f"V_safe={meta['V_safe']:.3f} V (headroom ref), "
        f"V_shrink={meta.get('V_shrink_trigger', meta['V_safe']):.3f} V "
        f"(= V_safe − {meta.get('V_deadband_mv', 0.0):.0f} mV, single trigger), "
        f"grow_step=+{meta['grow_step']} -> "
        f"V_ac={meta['Vac_grow_v']*1e3:.1f} mV/grow\n"
        f"shrinks={s['n_shrinks']}, grows={s['n_grows']}  |  "
        f"V_min violations (control phase)={s['n_violations']}"
        f" / total={s.get('n_violations_total', s['n_violations'])}  |  "
        f"total {s['total_flops']/1e12:.2f} TFLOPs in {s['total_s']:.1f} s "
        f"({s['avg_tflops']:.3f} TFLOP/s)  |  "
        f"M[min/avg/max]={s['M_min']}/{s['M_avg']:.0f}/{s['M_max']} (post-ramp)  |  "
        f"sw latency avg={s['avg_latency_us']:.1f} us, "
        f"max={s['max_latency_us']:.1f} us"
    )
    if ramp_up_s > 0:
        title = (
            f"Ramp {ramp_up_s:.1f} s  ->  closed-loop PDN feedback  |  "
            + title
        )
    if title_extra:
        title += "\n" + title_extra
    fig.suptitle(title, fontsize=9.5,
                 color="tab:red" if not throttle_on else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved stacked timeline -> {path}")



# Throttle-ON vs throttle-OFF comparison plots (--compare output).

def _compare_fig(meta_on: dict, figsize=(12, 5)):
    """Open a single-axes compare figure with ramp-phase shading."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ramp_up_s = float(meta_on.get("ramp_up_s", 0.0))
    if ramp_up_s > 0:
        ax.axvspan(0.0, ramp_up_s, color="tab:grey", alpha=0.10, zorder=0)
        ax.axvline(ramp_up_s, color="tab:grey", linestyle=":", linewidth=1.0)
    return fig, ax, ramp_up_s


def _compare_title(metric: str, meta_on: dict, s_on: dict, s_off: dict) -> str:
    return (
        f"{metric} — throttle ON vs OFF — kernel={meta_on['kernel']}  "
        f"R={meta_on['R']*1e3:.2f} mΩ, L={meta_on['L']*1e9:.1f} nH, "
        f"C={meta_on['C']*1e3:.3f} mF\n"
        f"V_min={meta_on['V_min']:.3f} V, V_safe={meta_on['V_safe']:.3f} V  |  "
        f"TFLOP/s (ON/OFF)={s_on['avg_tflops']:.3f}/{s_off['avg_tflops']:.3f}  |  "
        f"V_min violations (ON/OFF)={s_on['n_violations']}/{s_off['n_violations']}"
    )


def plot_compare_M(
    tl_on: List[Tick],  ev_on:  List[Event], meta_on:  dict,
    tl_off: List[Tick], ev_off: List[Event], meta_off: dict,
    path: str,
) -> None:
    """Batch-size M vs time, throttle ON overlaid on throttle OFF."""
    if not tl_on or not tl_off:
        print("No data to plot (M compare).")
        return

    t_on,  M_on,  *_ = _arrays(tl_on)
    t_off, M_off, *_ = _arrays(tl_off)
    s_on  = _summarize(tl_on,  ev_on,  meta_on)
    s_off = _summarize(tl_off, ev_off, meta_off)

    fig, ax, _ = _compare_fig(meta_on)
    ax.step(t_on,  M_on,  where="post", color="tab:blue", linewidth=1.8,
            label=(f"THROTTLE ON  — M_avg={s_on['M_avg']:.0f}, "
                   f"shrinks={s_on['n_shrinks']}, grows={s_on['n_grows']}"))
    ax.step(t_off, M_off, where="post", color="tab:red",  linewidth=1.5,
            alpha=0.75,
            label=f"THROTTLE OFF — M_avg={s_off['M_avg']:.0f}")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Batch size M")
    ax.set_ylim(0, meta_on.get("M_max", _M_MAX) * 1.10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(_compare_title("Batch size M", meta_on, s_on, s_off),
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved batch-size comparison -> {path}")


def plot_compare_power(
    tl_on: List[Tick],  ev_on:  List[Event], meta_on:  dict,
    tl_off: List[Tick], ev_off: List[Event], meta_off: dict,
    path: str,
) -> None:
    """NVML-measured GPU power vs time, throttle ON overlaid on throttle OFF."""
    nv_on  = meta_on.get("nvml_trace",  []) or []
    nv_off = meta_off.get("nvml_trace", []) or []
    if not nv_on or not nv_off:
        print("No NVML power data to plot (power compare).")
        return

    t_on,  p_on  = zip(*nv_on)
    t_off, p_off = zip(*nv_off)
    p_on_arr, p_off_arr = np.asarray(p_on), np.asarray(p_off)
    s_on  = _summarize(tl_on,  ev_on,  meta_on)
    s_off = _summarize(tl_off, ev_off, meta_off)

    fig, ax, _ = _compare_fig(meta_on)
    ax.plot(t_on,  p_on_arr,  color="tab:blue", linewidth=1.2,
            label=(f"THROTTLE ON  — mean={p_on_arr.mean():.1f} W, "
                   f"max={p_on_arr.max():.1f} W"))
    ax.plot(t_off, p_off_arr, color="tab:red",  linewidth=1.2, alpha=0.8,
            label=(f"THROTTLE OFF — mean={p_off_arr.mean():.1f} W, "
                   f"max={p_off_arr.max():.1f} W"))

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("GPU power [W]  (NVML)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    extra = (f"Calibration: P_idle={meta_on['P_idle_w']:.0f}±"
             f"{meta_on.get('P_idle_std_w', 0.0):.1f} W, "
             f"P_full={meta_on['P_full_w']:.0f}±"
             f"{meta_on.get('P_full_std_w', 0.0):.1f} W")
    fig.suptitle(_compare_title("GPU power (NVML)", meta_on, s_on, s_off)
                 + "\n" + extra, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved NVML-power comparison -> {path}")


def plot_compare_vload(
    tl_on: List[Tick],  ev_on:  List[Event], meta_on:  dict,
    tl_off: List[Tick], ev_off: List[Event], meta_off: dict,
    path: str,
) -> None:
    """V_load vs time, throttle ON overlaid on throttle OFF, with the
    V_safe / V_shrink / V_min reference lines and V_min violation shading."""
    if not tl_on or not tl_off:
        print("No data to plot (V_load compare).")
        return

    t_on,  _, _, Vw_on,  _ = _arrays(tl_on)
    t_off, _, _, Vw_off, _ = _arrays(tl_off)
    s_on  = _summarize(tl_on,  ev_on,  meta_on)
    s_off = _summarize(tl_off, ev_off, meta_off)

    V_min    = meta_on["V_min"]
    V_safe   = meta_on["V_safe"]
    V_shrink = meta_on.get("V_shrink_trigger", V_safe)
    db_mv    = meta_on.get("V_deadband_mv", 0.0)

    fig, ax, _ = _compare_fig(meta_on)
    ax.plot(t_on,  Vw_on,  color="tab:blue", linewidth=1.3,
            label=(f"V_load THROTTLE ON  — min={float(Vw_on.min()):.3f} V, "
                   f"viol={s_on['n_violations']}"))
    ax.plot(t_off, Vw_off, color="tab:red",  linewidth=1.3, alpha=0.8,
            label=(f"V_load THROTTLE OFF — min={float(Vw_off.min()):.3f} V, "
                   f"viol={s_off['n_violations']}"))

    ax.axhline(V_safe, color="tab:orange", linestyle="--", linewidth=1.2,
               label=f"V_safe ({V_safe:.3f} V)")
    if V_shrink < V_safe - 1e-6:
        ax.axhline(V_shrink, color="tab:orange", linestyle=":", linewidth=1.0,
                   label=f"V_shrink ({V_shrink:.3f} V = V_safe − {db_mv:.0f} mV)")
    ax.axhline(V_min, color="tab:red", linestyle="--", linewidth=1.0,
               label=f"V_min ({V_min:.3f} V)")

    ax.fill_between(t_off, Vw_off, V_min, where=(Vw_off < V_min),
                    color="red", alpha=0.25)
    ax.fill_between(t_on,  Vw_on,  V_min, where=(Vw_on  < V_min),
                    color="red", alpha=0.40)

    V_reach_lo = meta_on.get("V_load_min_dc", V_min)
    V_reach_hi = meta_on.get("V_load_max_dc", V_safe)
    v_lo = min(float(Vw_on.min()), float(Vw_off.min()),
               V_min, V_reach_lo, V_shrink) - 0.005
    v_hi = max(float(Vw_on.max()), float(Vw_off.max()),
               V_safe, V_reach_hi) + 0.005
    ax.set_ylim(v_lo, v_hi)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("V_load [V]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, ncol=2)

    fig.suptitle(_compare_title("V_load", meta_on, s_on, s_off), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved V_load comparison -> {path}")


# CLI

def _argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SlewGuard closed-loop controller")
    p.add_argument("--kernel", type=str,   default="tiled_gemm")
    p.add_argument("--list-kernels", action="store_true")
    p.add_argument("--R", type=float, default=5.0e-4)
    p.add_argument("--L", type=float, default=1.5e-8)
    p.add_argument("--C",  type=float, default=1.0e-3)
    p.add_argument("--V-min", type=float, default=0.95)
    p.add_argument("--total-seconds", type=float, default=30.0)
    p.add_argument("--tick-us", type=float, default=_DEFAULT_TICK_US)
    p.add_argument("--poll-us", type=float, default=_DEFAULT_POLL_US)
    p.add_argument("--grow-period-us", type=float,
                   default=_DEFAULT_GROW_PERIOD_US,
                   help="Min wall time between grows [us]. Default = tau_env "
                        "(200 us): each grow's V_ac decays before the next "
                        "so V_ac bursts can't stack past V_safe−V_min.")
    p.add_argument("--shrink-period-us", type=float,
                   default=_DEFAULT_SHRINK_PERIOD_US,
                   help="Min wall time between shrinks [us]. Default 1000 "
                        "(= 5·tau_env): one LC ring -> one shrink, realistic "
                        "driver-layer throttle latency.")
    p.add_argument("--deadband-mv", type=float,
                   default=_V_SAFE_HYSTERESIS_MV,
                   help=f"V_safe -> V_shrink hysteresis deadband [mV]. "
                        f"Shrinks fire only when V_load < V_safe − "
                        f"deadband_mv. Auto-clamped so V_shrink ≥ V_min+"
                        f"3 mV. 0 = single-threshold bang-bang at V_safe. "
                        f"Default = {_V_SAFE_HYSTERESIS_MV:.0f}.")

    p.add_argument("--no-throttle", action="store_true",
                   help="Disable the controller — after the ramp M stays "
                        "at M_MAX for the whole run (UNTHROTTLED BASELINE).")
    
    p.add_argument("--pdn-noise-scale", type=float, default=_DEFAULT_PDN_NOISE_SCALE) 
    
    p.add_argument("--ramp-up-s", type=float,
                   default=_DEFAULT_RAMP_UP_S,
                   help=f"Open-loop warm-up ramp duration [s]. M is driven "
                        f"linearly {_M_MIN}->{_M_MAX} during this window; "
                        f"controller observes only. 0 = start closed-loop "
                        f"immediately. Default = {_DEFAULT_RAMP_UP_S:.1f}.")
    p.add_argument("--compare", action="store_true",
                   help="Run twice (throttle ON, then OFF) back-to-back and "
                        "emit three comparison plots (M, NVML power, V_load). "
                        "To compare different PDN configurations, invoke "
                        "--compare separately with different --R/--L/--C.")
    p.add_argument("--out-dir", type=str, default=".",
                   help="Directory for output CSV and PNG files. Created if it "
                        "does not exist. Default: current directory.")
    p.add_argument("--workload", type=str, default="gemm",
                   choices=["gemm", "resnet"],
                   help="Workload backend: 'gemm' (default) uses the CUDA GEMM "
                        "kernel registry; 'resnet' runs ResNet-18 inference.")
    p.add_argument("--M-min", type=int, default=None,
                   help="Minimum batch size M (default: 64 for gemm, 1 for resnet).")
    p.add_argument("--M-max", type=int, default=None,
                   help="Maximum batch size M (default: 512 for gemm, 16 for resnet).")
    return p


def _print_summary(label: str, s: dict) -> None:
    print(f"  {label:<36s}  "
          f"{s['total_flops']/1e12:6.2f} TFLOPs  "
          f"{s['avg_tflops']:.3f} TFLOP/s  "
          f"shrinks={s['n_shrinks']:>4d}  "
          f"grows={s['n_grows']:>4d}  "
          f"viol={s['n_violations']:>4d}  M_avg={s['M_avg']:.0f}")


def main() -> None:
    args = _argparser().parse_args()

    if args.list_kernels:
        for idx, name in list_kernels():
            print(f"  [{idx}] {name}")
        return

    _resnet_defaults = {"M_min": 1, "M_max": 16}
    _gemm_defaults   = {"M_min": _M_MIN, "M_max": _M_MAX}
    _wl_defaults = _resnet_defaults if args.workload == "resnet" else _gemm_defaults
    M_min = args.M_min if args.M_min is not None else _wl_defaults["M_min"]
    M_max = args.M_max if args.M_max is not None else _wl_defaults["M_max"]

    kw = dict(
        V_min=args.V_min, total_seconds=args.total_seconds,
        kernel_name=args.kernel, tick_us=args.tick_us,
        poll_us=args.poll_us, grow_period_us=args.grow_period_us,
        shrink_period_us=args.shrink_period_us,
        throttle_enabled=(not args.no_throttle),
        ramp_up_s=args.ramp_up_s,
        deadband_mv=args.deadband_mv,
        pdn_noise_scale=args.pdn_noise_scale,
        workload=args.workload,
        M_min=M_min,
        M_max=M_max,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    pfx = os.path.join(args.out_dir, _OUT_PREFIX)

    if args.compare:
        kw_on  = dict(kw); kw_on["throttle_enabled"]  = True
        kw_off = dict(kw); kw_off["throttle_enabled"] = False

        print("\n=== RUN 1/2: THROTTLE ENABLED ===")
        tl_on, ev_on, m_on = run_live(R=args.R, L=args.L, C=args.C, **kw_on)
        write_csv(tl_on, f"{pfx}_throttle.csv")

        print("\n=== RUN 2/2: UNTHROTTLED BASELINE ===")
        tl_off, ev_off, m_off = run_live(R=args.R, L=args.L, C=args.C, **kw_off)
        write_csv(tl_off, f"{pfx}_unthrottled.csv")

        plot_compare_M(    tl_on, ev_on, m_on, tl_off, ev_off, m_off,
                           f"{pfx}_compare_M.png")
        plot_compare_power(tl_on, ev_on, m_on, tl_off, ev_off, m_off,
                           f"{pfx}_compare_power.png")
        plot_compare_vload(tl_on, ev_on, m_on, tl_off, ev_off, m_off,
                           f"{pfx}_compare_vload.png")

        s_on  = _summarize(tl_on,  ev_on,  m_on)
        s_off = _summarize(tl_off, ev_off, m_off)
        print("\n=== Throttle comparison ===")
        _print_summary("throttle ON",         s_on)
        _print_summary("throttle OFF (base)", s_off)
        dT = 100*(s_off['avg_tflops']-s_on['avg_tflops']) / max(s_on['avg_tflops'], 1e-9)
        dV = s_off['n_violations'] - s_on['n_violations']
        print(f"  enabling throttle costs {-dT:+.1f}% TFLOP/s "
              f"and eliminates {dV} V_min violations")
        return

    # Single run
    tl, ev, meta = run_live(R=args.R, L=args.L, C=args.C, **kw)
    write_csv(tl, f"{pfx}.csv")
    plot_stacked(tl, ev, meta, f"{pfx}.png")

    s = _summarize(tl, ev, meta)
    print()
    print(f"kernel={meta['kernel']}  records={len(tl)}  "
          f"iters={meta['n_iters']}  launches={meta['n_launches']}")
    print(f"M[min/avg/max]={s['M_min']}/{s['M_avg']:.0f}/{s['M_max']}  "
          f"total={s['total_flops']/1e12:.2f} TFLOPs  "
          f"avg={s['avg_tflops']:.3f} TFLOP/s")
    print(f"shrinks={s['n_shrinks']}  grows={s['n_grows']}  "
          f"violations={s['n_violations']}  "
          f"sw latency avg={s['avg_latency_us']:.1f} us  "
          f"max={s['max_latency_us']:.1f} us")
    print(f"tick={meta['tick_us']:.1f} us  achieved={meta['achieved_tick_us']:.1f} us")


if __name__ == "__main__":
    main()
