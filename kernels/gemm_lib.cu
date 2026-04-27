// SlewGuard

// Persistent device buffers + a small kernel registry used so multipe kernels share this buffer.

// Decoupled memory alloc and event creation from actual kernel math to avoid polluting the latency reported to controller.

// Adding kernels:
//   1. Implement `static int run_<name>(int M, float *out_ms)` that launches
//      kernel using the shared persistent buffers and returns the CUDA-event measured kernel time in *out_ms.
//   2. Append one row to g_kernels[].

// `kernel_select_by_name()`: used by python to run kernels


#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

#define CUDA_CHECK(call) do {                                                  \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                        \
                #call, __FILE__, __LINE__, cudaGetErrorString(err__));         \
        return -1;                                                             \
    }                                                                          \
} while (0)


// Shared persistent buffers for matrices
// A: (M_max x K)
// B: (K x N)
// C: (M_max x N)

static float *g_dA = nullptr;
static float *g_dB = nullptr;
static float *g_dC = nullptr;
static int g_Mmax = 0;
static int g_N = 0;
static int g_K = 0;
static cudaEvent_t  g_start;
static cudaEvent_t  g_stop;
static int g_initialized = 0;


// Timing helper

template <typename Launch>
static int timed_launch(Launch launch, float *out_ms) {
    CUDA_CHECK(cudaEventRecord(g_start, 0));
    launch();
    CUDA_CHECK(cudaEventRecord(g_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(g_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, g_start, g_stop));
    if (out_ms) *out_ms = ms;
    return 0;
}


// Kernel #1: naive GEMM 

__global__ void gemm_naive_kernel(const float *A, const float *B, float *C,
                                  int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

static int run_naive_gemm(int M, float *out_ms) {
    if (!g_initialized || M > g_Mmax) return -1;
    dim3 threads(16, 16);
    dim3 blocks((g_N + 15) / 16, (M + 15) / 16);
    return timed_launch([&]() {
        gemm_naive_kernel<<<blocks, threads>>>(g_dA, g_dB, g_dC, M, g_N, g_K);
    }, out_ms);
}


// Kernel #2: tiled GEMM + fused GELU activation epilogue.
// GELU(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))

#define TILE 16

__device__ __forceinline__ float gelu_approx(float x) {
    // sqrt(2/pi) = 0.7978845608f
    float x3 = x * x * x;
    float u  = 0.7978845608f * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(u));
}

__global__ void gemm_tiled_kernel(const float *A, const float *B, float *C,
                                  int M, int N, int K) {
    __shared__ float As[TILE][TILE+1];
    __shared__ float Bs[TILE][TILE+1];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;
    int n_tiles = (K + TILE - 1) / TILE;

    // Phase A: tiled matmul (memory + compute mix).
    for (int t = 0; t < n_tiles; t++) {
        int Acol = t * TILE + threadIdx.x;
        int Brow = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
			(row < M && Acol < K) ? A[row * K + Acol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
			(Brow < K && col < N) ? B[Brow * N + col] : 0.0f;

        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    // Phase B: fused GELU epilogue (sharp di/dt).
    if (row < M && col < N)
        C[row * N + col] = gelu_approx(sum);
}

static int run_tiled_gemm(int M, float *out_ms) {
    if (!g_initialized || M > g_Mmax) return -1;
    dim3 threads(TILE, TILE);
    dim3 blocks((g_N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    return timed_launch([&]() {
        gemm_tiled_kernel<<<blocks, threads>>>(g_dA, g_dB, g_dC, M, g_N, g_K);
    }, out_ms);
}


// Kernel registry. All kernels can share these.

typedef int (*kernel_run_fn)(int M, float *out_ms);

struct KernelEntry {
    const char   *name;
    kernel_run_fn run;
};

static const KernelEntry g_kernels[] = {
    {"naive_gemm", run_naive_gemm},
    {"tiled_gemm", run_tiled_gemm},   // matmul + fused GELU
    // Add new kernels here.  The controller drives the PDN model from whichever kernel is active
};
static const int g_num_kernels = (int)(sizeof(g_kernels) / sizeof(g_kernels[0]));
static int       g_active      = 0;


// Public registry API.

extern "C" int kernel_count(void) {
    return g_num_kernels;
}

extern "C" const char *kernel_name(int idx) {
    if (idx < 0 || idx >= g_num_kernels) return nullptr;
    return g_kernels[idx].name;
}

extern "C" int kernel_active(void) {
    return g_active;
}

extern "C" int kernel_select_by_index(int idx) {
    if (idx < 0 || idx >= g_num_kernels) return -1;
    g_active = idx;
    return 0;
}

extern "C" int kernel_select_by_name(const char *name) {
    if (!name) return -1;
    for (int i = 0; i < g_num_kernels; i++) {
        if (strcmp(name, g_kernels[i].name) == 0) {
            g_active = i;
            return i;
        }
    }
    return -1;
}


// Public buffer / dispatch API.

extern "C" int gemm_init(int M_max, int N, int K) {
    if (g_initialized) return 0;
    g_Mmax = M_max;
    g_N    = N;
    g_K    = K;

    CUDA_CHECK(cudaMalloc(&g_dA, (size_t)M_max * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_dB, (size_t)K     * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_dC, (size_t)M_max * N * sizeof(float)));
    CUDA_CHECK(cudaEventCreate(&g_start));
    CUDA_CHECK(cudaEventCreate(&g_stop));

    g_initialized = 1;
    return 0;
}

extern "C" int gemm_upload_A(const float *h_A, int M) {
    if (!g_initialized || M > g_Mmax) return -1;
    CUDA_CHECK(cudaMemcpy(g_dA, h_A,
                          (size_t)M * g_K * sizeof(float),
                          cudaMemcpyHostToDevice));
    return 0;
}

extern "C" int gemm_upload_B(const float *h_B) {
    if (!g_initialized) return -1;
    CUDA_CHECK(cudaMemcpy(g_dB, h_B,
                          (size_t)g_K * g_N * sizeof(float),
                          cudaMemcpyHostToDevice));
    return 0;
}

extern "C" int gemm_run(int M, float *out_ms) {
    if (!g_initialized || M > g_Mmax) return -1;
    return g_kernels[g_active].run(M, out_ms);
}

extern "C" int gemm_destroy(void) {
    if (!g_initialized) return 0;
    if (g_dA) cudaFree(g_dA);
    if (g_dB) cudaFree(g_dB);
    if (g_dC) cudaFree(g_dC);
    cudaEventDestroy(g_start);
    cudaEventDestroy(g_stop);
    g_dA = g_dB = g_dC = nullptr;
    g_Mmax = g_N = g_K = 0;
    g_initialized = 0;
    return 0;
}
