// 05_float4_vectorized_b.cu
// Variant using float4 loads on A and pre-transposed B_t (N x K).
// Pre-transpose + allocations are handled by shared runner `run_once_vectorized_b` (not timed).

#include "common/utils.cuh"
__global__ void vectorized_kernel(
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B_t,
    float beta,
    float* __restrict__ C,
    int M, int K, int N
)
{   

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x; // one column per block; threads.x used to stride k
    int lane = threadIdx.x & 31;

    if (row >= M || col >= N) return;
    
    float sum = 0.0f;

    // // ORIGINAL: 01_naive version = scalar loads = slow
    // for (int k = 0; k < K; ++k){
    //     sum += A[row * K + k] * B[k * N + col];
    // }


    // //handle the whole row_of_A * column_of_B FMA without the tail (remainder after dividing their lengths by 4)
    // for (int k = 0; k + 3 < K; k+=4){
    //     ////THIS IS THE MAGIC - one thread loads 4 floats at once from global memory (only for A, which is row-wise consecutive)
    //     float4 a = ((const float4*)&A[row * K + k])[0];
    //     // With B pre-transposed to B_t (N x K), each thread's column 'col' is contiguous in k.
    //     // Load 4 consecutive floats from B_t for the same col, but now access is non-coalesced! (throughput of ca 550 gflops)
    //     // with the current “one thread = one C[row,col]” mapping, you can’t have both per‑thread contiguous float4 and warp‑coalesced accesses for B at the same time.
    //     //float4 b4 = ((const float4*)&B_t[col * K + k])[0];


    // Remapped lanes: warp fixes 'col' and advances k in lockstep.
    // Row 45 explanation: each lane l loads B_t[col*K + (k_base + 4*l)..(k_base + 4*l + 3)],
    // so across lanes the addresses are contiguous → coalesced. Each thread accumulates its row's partial sum.
    for (int k_base = 0; k_base + 4*32 - 1 < K; k_base += 4*32){
        int k = k_base + 4*lane;
        float4 a4 = ((const float4*)&A[row * K + k])[0];
        float4 b4 = ((const float4*)&B_t[col * K + k])[0];
        sum += a4.x * b4.x + a4.y * b4.y + a4.z * b4.z + a4.w * b4.w;
    }

    //handle the tail 
    for (int k=((K/(4*32))*(4*32)); k < K; ++k){
        sum += A[row * K + k] * B_t[col * K + k];
    }

    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

// Core GEMM launcher: assumes B_t (N x K) already prepared.
void launch_vectorized_core(float alpha, const float* A, const float* B_t, float beta, float* C, int M, int K, int N){
    dim3 threads(32,16); // 32 lanes advance k in lockstep per warp; 16 rows per block
    dim3 blocks((N + 1 - 1) / 1, (M + threads.y - 1) / threads.y); // one column per block in x
    vectorized_kernel<<<blocks, threads>>>(alpha, A, B_t, beta, C, M, K, N);
    CHECK_CUDA(cudaGetLastError());
}

int main(){
    std::cout << "===05_float4_vectorized_b (pre-transpose B off timing)===\n";
    for (int s : TEST_SIZE){
        run_once_vectorized_b("05_float4_vectorized_b", launch_vectorized_core, s, s, s,
                              /*blockDim=*/32*16, /*dynamicSmemBytes=*/0,
                              (const void*)vectorized_kernel);
    }
    return 0;
}

// nvcc -O3 -arch=sm_86 05_float4_vectorized_b.cu -o 05_float4_vectorized_b && ./05_float4_vectorized_b | tee 05_float4_vectorized_b.txt