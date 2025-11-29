// 06_autotune_tiles.cu
// Shared-memory tiled GEMM variants for TILE sizes {8,16,32}.
// Uses common `run_gemm_test` (alloc/copy + timing) for consistency.
// Autotune concept: compare GFLOPS across tile sizes; external script can pick best.

#include "common/utils.cuh"

template<int TILE>
__global__ void tiled_kernel(
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C,
    int M, int K, int N
)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row >= M or col >= N) return;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    __shared__ float A_tile[TILE][TILE];
    __shared__ float B_tile[TILE][TILE+1];

    float sum = 0.0f;

    for (int tile_k = 0; tile_k < K; tile_k += TILE)
    {
        A_tile[ty][tx] = (row < M && tile_k + tx < K) ? A[row * K + (tile_k + tx)] : 0.0f;
        B_tile[ty][tx] = ((tile_k + ty) < K && col < N) ? B[(tile_k + ty) * N + col] : 0.0f;
        __syncthreads();
        
        for (int k = 0; k < TILE; ++k)
            sum += A_tile[ty][k] * B_tile[k][tx];
        __syncthreads();
    }
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

// Launch wrappers matching run_gemm_test signature
void launch_tiled_8 (float alpha,const float* A,const float* B,float beta,float* C,int M,int K,int N){
    dim3 threads(8,8); dim3 blocks((N+7)/8,(M+7)/8); tiled_kernel<8><<<blocks,threads>>>(alpha,A,B,beta,C,M,K,N); CHECK_CUDA(cudaGetLastError()); }
void launch_tiled_16(float alpha,const float* A,const float* B,float beta,float* C,int M,int K,int N){
    dim3 threads(16,16); dim3 blocks((N+15)/16,(M+15)/16); tiled_kernel<16><<<blocks,threads>>>(alpha,A,B,beta,C,M,K,N); CHECK_CUDA(cudaGetLastError()); }
void launch_tiled_32(float alpha,const float* A,const float* B,float beta,float* C,int M,int K,int N){
    dim3 threads(32,32); dim3 blocks((N+31)/32,(M+31)/32); tiled_kernel<32><<<blocks,threads>>>(alpha,A,B,beta,C,M,K,N); CHECK_CUDA(cudaGetLastError()); }

int main(){
    std::cout << "===06_autotune_tiles (run_gemm_test)===\n";
    for (int s : TEST_SIZE){
        run_gemm_test("06_tiles_T8",  launch_tiled_8,  s, s, s, 8*8, 0,   (const void*)tiled_kernel<8>);
        run_gemm_test("06_tiles_T16", launch_tiled_16, s, s, s, 16*16, 0, (const void*)tiled_kernel<16>);
        run_gemm_test("06_tiles_T32", launch_tiled_32, s, s, s, 32*32, 0, (const void*)tiled_kernel<32>);
    }
    return 0;
}

//nvcc -O3 -arch=sm_86 06_autotune_tiles.cu -o 06_autotune_tiles && ./06_autotune_tiles | tee 06_autotune_tiles.txt