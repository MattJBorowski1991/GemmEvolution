// 02_anti_coalesced.cu
// Anti-coalesced access pattern for B: B is passed transposed as B_t (N x K). Same thread layout, only indexing changed.
// Warp threads vary 'col'; for fixed k they read B_t[col * K + k] with stride K floats (4*K bytes).
// Large K forces each lane into distinct cache lines, reducing memory coalescing efficiency.
// ~8–10× speedup over naïve → ~1.5–2 TFLOPS on modern GPUs.

#include "common/utils.cuh"
#include "common/metrics.cuh"

__global__ void anti_coalesced_kernel(
    float alpha,
    const float* __restrict__ A,        // M x K
    const float* __restrict__ B_t,      // N x K (transposed B)
    float beta,
    float* __restrict__ C,              // M x N (output)
    int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        // For fixed k: lane addresses differ by K floats (anti-coalesced on B_t).
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B_t[col * K + k];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

void launch_anti_coalesced(float alpha, const float* A, const float* B_t, float beta, float* C, int M, int K, int N)
{
    dim3 threads(16,16);
    dim3 blocks((N + threads.x - 1)/threads.x,(M + threads.y - 1)/threads.y);

    anti_coalesced_kernel<<<blocks,threads>>>(alpha, A, B_t, beta, C, M, K, N);

    CHECK_CUDA(cudaGetLastError());
}

int main(){
    std::cout << "=== 02_anti_coalesced (B transposed) === \n";
    for (int s : TEST_SIZE) {
        run_gemm_test("02_anti_coalesced", launch_anti_coalesced, s, s, s,
                      /*blockDim=*/16*16, /*dynamicSmemBytes=*/0,
                      (const void*)anti_coalesced_kernel);
    }
    return 0;
}

// nvcc -O3 -arch=sm_86 02_anti_coalesced.cu -o 02_anti_coalesced && ./02_anti_coalesced | tee 02_anti_coalesced.txt
