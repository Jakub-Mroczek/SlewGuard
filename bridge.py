import ctypes
import numpy as np
import os

# Load the shared library
lib_path = os.path.join(os.path.dirname(__file__), "lib", "libgemm.so")
lib = ctypes.CDLL(lib_path)

# Tell ctypes the function signature
lib.run_gemm.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # A
    ctypes.POINTER(ctypes.c_float),  # B
    ctypes.POINTER(ctypes.c_float),  # C
    ctypes.c_int,                    # M
    ctypes.c_int,                    # N
    ctypes.c_int,                    # K
]
lib.run_gemm.restype = None

def run_gemm(A, B, batch_size=None):
    """
    Run GEMM: C = A @ B
    A: (M, K) numpy float32
    B: (K, N) numpy float32
    batch_size: if set, only process the first batch_size rows of A
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "A columns must match B rows"

    if batch_size is not None:
        A = A[:batch_size]
        M = batch_size

    C = np.zeros((M, N), dtype=np.float32)

    lib.run_gemm(
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(K),
    )
    return C


if __name__ == "__main__":
    N = 1024
    A = np.ones((N, N), dtype=np.float32)
    B = np.ones((N, N), dtype=np.float32)

    for batch_size in [64, 128, 256, 512, 1024]:
        C = run_gemm(A, B, batch_size=batch_size)
        print(f"batch_size={batch_size:4d}  C[0][0]={C[0][0]:.1f}  (expected {float(N):.1f})")
