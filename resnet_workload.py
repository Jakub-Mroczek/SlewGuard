"""ResNet-18 inference workload for SlewGuard controller.

Same run(M) -> (rc, elapsed_ms) interface as gemm_run.
The controllers kernel-worker thread can swap in this workload with minimal changes.
Batch size M is the control knob. Power scales with M just as matrix rows do in the GEMM workloads.

Typical M range on a V100: M_MIN=1, M_MAX=16 (each image is 3×224×224).
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torchvision.models as tvm

# ResNet-18 FLOPs at 224×224 (MACs×2, standard estimate).
_RESNET18_FLOPS_PER_IMAGE = 1_820_000_000.0

_DEFAULT_M_MIN = 1
_DEFAULT_M_MAX = 16
_INPUT_C = 3
_INPUT_H = 224
_INPUT_W = 224


class ResNetWorkload:
    """ResNet-18 inference workload.

    Usage::
        wl = ResNetWorkload(M_max=16)
        rc, ms = wl.run(8)   # batch-8 inference
        wl.destroy()
    """

    def __init__(self, M_max: int = _DEFAULT_M_MAX) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device required for ResNetWorkload")

        self.M_max = M_max
        self.M_min = _DEFAULT_M_MIN
        self._device = torch.device("cuda", 0)

        self._model: nn.Module = tvm.resnet18(weights=None)
        self._model.eval()
        self._model.to(self._device)

        # Preallocate max-batch input once. Slice to [:M] each run.
        self._x = torch.randn(
            M_max, _INPUT_C, _INPUT_H, _INPUT_W,
            device=self._device, dtype=torch.float32,
        )

        # CUDA events for kernel timing
        self._ev_start = torch.cuda.Event(enable_timing=True)
        self._ev_stop  = torch.cuda.Event(enable_timing=True)

        # Warm up: two passes so CUDA graphs / cuDNN benchmarking settle.
        with torch.no_grad():
            self._model(self._x)
            self._model(self._x)
        torch.cuda.synchronize()

    def run(self, M: int) -> tuple[int, float]:
        """Run ResNet-18 inference at batch size M.

        Returns (rc, elapsed_ms) where rc=0 on success, rc=-1 on error.
        Blocks until the GPU kernel finishes (releases GIL during sync).
        """
        if M < 1 or M > self.M_max:
            return -1, 0.0

        x = self._x[:M]
        try:
            self._ev_start.record()
            with torch.no_grad():
                self._model(x)
            self._ev_stop.record()
            # elapsed_time() calls cudaEventSynchronize, releasing the GIL.
            torch.cuda.current_stream().synchronize()
            elapsed_ms = self._ev_start.elapsed_time(self._ev_stop)
        except Exception:
            return -1, 0.0

        return 0, elapsed_ms

    def flops(self, M: int) -> float:
        """Approximate FLOPs for a batch-M forward pass."""
        return M * _RESNET18_FLOPS_PER_IMAGE

    def destroy(self) -> None:
        del self._x
        del self._model
        torch.cuda.empty_cache()
