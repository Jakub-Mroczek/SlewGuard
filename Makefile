NVCC     = nvcc
ARCH     = -arch=sm_70
CFLAGS   = -O2

all: gemm_naive lib/libgemm.so

gemm_naive: kernels/gemm_naive.cu
	$(NVCC) $(ARCH) $(CFLAGS) -o gemm_naive kernels/gemm_naive.cu

lib/libgemm.so: kernels/gemm_lib.cu
	$(NVCC) $(ARCH) $(CFLAGS) --compiler-options '-fPIC' -shared -o lib/libgemm.so kernels/gemm_lib.cu

clean:
	rm -f gemm_naive lib/libgemm.so
