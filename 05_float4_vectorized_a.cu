// 05_float4_vectorized.cu
// From pure 01_naive -> ONLY added float4 vector loads
// expected 15-20x speedup over 01_naive -> 3-5 TFLOPS on modern GPU

#include "common/utils.cuh"

__global__ void vectorized_kernel(
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C,
    int M, int K, int N
)
{   

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;
    
    float sum = 0.0f;

    // // ORIGINAL: 01_naive version = scalar loads = slow
    // for (int k = 0; k < K; ++k){
    //     sum += A[row * K + k] * B[k * N + col];
    // }

    //handle the whole row_of_A * column_of_B FMA without the tail (remainder after dividing their lengths by 4)
    for (int k = 0; k + 3 < K; k+=4){
        ////THIS IS THE MAGIC - one thread loads 4 floats at once from global memory (only for A, which is row-wise consecutive)
        // note however that thread access to A is non-coalesced (K-strided)
        float4 a = ((const float4*)&A[row * K + k])[0];
        // TODO pre-transpose B - better than scalar
        // B is column-wise (strided), so use scalar loads - otherwise you wouldn't load 4 consecutive elements from memory
        float b0 = B[(k+0) * N + col];
        float b1 = B[(k+1) * N + col];
        float b2 = B[(k+2) * N + col];
        float b3 = B[(k+3) * N + col];

        sum += a.x * b0 + a.y * b1 + a.z * b2 + a.w * b3;
    }

    //handle the tail 
    for (int k=(K & ~3); k < K; ++k){ // (K % ~3) = remove the last two bits from K = round down to the nearest multiple of 4. e.g. (1001 & ~3) = 1000.
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

void launch_vectorized(float alpha, const float* A, const float* B, float beta, float* C, int M, int K, int N){

    dim3 threads(16,16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    vectorized_kernel<<<blocks, threads>>>(alpha, A, B, beta, C, M, K, N);
    CHECK_CUDA(cudaGetLastError());
}

int main(){
    std::cout << "===05_float4_vectorized===\n";
    for (int s : TEST_SIZE)
        run_gemm_test("05_float4_vectorized", launch_vectorized, s, s, s,
                      /*blockDim=*/16*16, /*dynamicSmemBytes=*/0,
                      (const void*)vectorized_kernel);
    return 0;
}

// nvcc -O3 -arch=sm_86 05_float4_vectorized_a.cu -o 05_float4_vectorized_a && ./05_float4_vectorized_a | tee 05_float4_vectorized_a.txt