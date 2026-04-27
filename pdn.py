# PDN model

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PDN:
    R: float = 5.0e-4    # Ohm -- PDN series resistance
    L: float = 1.0e-8    # H -- PDN+package inductance
    C: float = 1.0e-3    # F -- bulk decoupling 
    V_min: float = 0.94  # can be changed in CLI

    # PDN time constant. Below tau_env seconds there is OS, and after tau_env there is steady state IR drop
    # Smoothens the DC current as well
    _TAU_ENV: float = 0.05

    def __post_init__(self) -> None:
        self.V_nom = 1.0
        self.Z_LC = math.sqrt(self.L / self.C)
        self.zeta = (self.R / 2.0) * math.sqrt(self.C / self.L)
        if self.zeta < 1.0:
            denom = math.sqrt(max(1.0 - self.zeta * self.zeta, 1e-12))
            self.overshoot = math.exp(-self.zeta * math.pi / denom)
        else:
            self.overshoot = 0.0
        self.reset()

    def reset(self) -> None:
        self._I_filt = 0.0
        self._init = False

    def step(self, power_w: float, dt: float) -> float:
        """Return V_load = V_nom - V_dc - V_ac for this sample."""
        I_raw = float(power_w) / self.V_nom
        if not self._init:
            # Pre-charge to steady state to avoid a startup inrush spike.
            self._I_filt = I_raw
            self._init = True
            return self.V_nom - I_raw * self.R

        # V_ac is driven by (I_raw - I_filt_OLD): the LC step size relative to the slow envelope.
        I_step = max(0.0, I_raw - self._I_filt)
        V_ac   = I_step * self.Z_LC * self.overshoot

        alpha       = dt / (self._TAU_ENV + dt)
        I_filt_new  = self._I_filt + alpha * (I_raw - self._I_filt)
        V_dc        = I_filt_new * self.R

        self._I_filt = I_filt_new
        return self.V_nom - V_dc - V_ac
