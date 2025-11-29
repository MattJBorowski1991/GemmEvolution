// 01_naive.cu
// One thread per C[i,j] element.
// 2D grid (16×16 threads), direct global memory access.
// Poor coalescing on B → memory-bound, ~150–200 GFLOPS.
// First working GPU version, ~25× faster than CPU.

#include "common/utils.cuh"

__global__ void naive_kernel(   float alpha,
                                const float* __restrict__ A,
                                const float* __restrict__ B,
                                float beta,
                                float* __restrict__ C,
                                int M, int K, int N)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N){
        float sum = 0.0f;
    for (int k=0; k<K; ++k){
        sum +=  A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

void launch_naive(float alpha, const float* A, const float* B, float beta, float* C, int M, int K, int N)
{
    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    naive_kernel<<<blocksPerGrid, threadsPerBlock>>>(alpha, A, B, beta, C, M, K, N);
    CHECK_CUDA(cudaGetLastError());
}

int main(){
    std::cout << "===01_naive===\n";
    for (int s : TEST_SIZE){
        run_gemm_test("01_naive", launch_naive, s, s, s,
                      /*blockDim=*/16*16, /*dynamicSmemBytes=*/0,
                      (const void*)naive_kernel);
    }
    return 0;
}

//nvcc -O3 -arch=sm_86 01_naive.cu -o 01_naive && ./01_naive | tee 01_naive.txt
//./01_naive