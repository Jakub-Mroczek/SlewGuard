NVCC     = nvcc
ARCH     = -arch=sm_70
CFLAGS   = -O2

all: lib/libgemm.so

lib/libgemm.so: kernels/gemm_lib.cu
	$(NVCC) $(ARCH) $(CFLAGS) --compiler-options '-fPIC' -shared -o lib/libgemm.so kernels/gemm_lib.cu

clean:
	rm -f lib/libgemm.so
