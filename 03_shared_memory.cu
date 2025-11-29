// 03_shared_memory_temp.cu
// Shared memory tiling: divide computation into BLOCKxBLOCK tiles.
// Load tiles of A and B into shared memory, compute partial sums, accumulate.
// Reduces global memory bandwidth by ~BLOCK/WARP factor.
// ~15–25× speedup over naïve → ~3–5 TFLOPS on modern GPUs.

#include "common/utils.cuh"

#define TILE_SIZE 16 //adjust based on 

__global__ void shared_memory_kernel(
    float alpha, 
    const float* __restrict__ A,    //M x K
    const float* __restrict__ B,    //K x N
    float beta, 
    float* __restrict__ C,          //M x N
    int M, int K, int N)
{
    // global thread ID = macro-coordinates within the output matrix
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    //macro-coordinates of tiles = (blockIdx.y, blockIdx.x) = tile-row of A, tile-column of B

    //local thread ID within a block = micro-coordinates within a tile = (ty, tx)
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    //declare and allocate shared memory for A_tile and B_tile - both of size TILE_SIZE x TILE_SIZE
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE+1];

    float sum = 0.0f;

    // iterate of tiles of A and B in the K dimension
    for (int tile_k = 0; tile_k < K; tile_k += TILE_SIZE)
    {   
        //// HERE IS WHERE THE MAGIC HAPPENS - LOADING TILES INTO SHARED MEMORY
        //firstly - the thread (ty, tx) loads data into a pair of matrices = (A_tile, B_tile)
        if (row < M && tile_k + tx < K) {
            A_tile[ty][tx] = A[row * K + (tile_k + tx)];
        } else {
            A_tile[ty][tx] = 0.0f;
        }

        if (tile_k + ty < K && col < N){
            B_tile[ty][tx] = B[(tile_k + ty) * N + col];
        } else {
            B_tile[ty][tx] = 0.0f;
        }
        
        __syncthreads(); // after this we have one pair of matrices filled up with data: (A_tile, B_tile)
        
        //secondly - the thread (ty, tx) calculates its own private "sum" value, which persists and accumulates throught the outer for loop steps
        for (int k = 0; k < TILE_SIZE && tile_k + k < K; ++k) {
            // reuse the data TILE_SIZE = 32 times from shared memory, avoiding slow access from global memory
            sum += A_tile[ty][k] * B_tile[k][tx];
            // alternative example for the line above, but when bank conflict would occur:
            // sum += A_tile[ty][k] * B_tile[k][k];
            // all 32 threads want to access B_tile[k][k] = 32-way conflict = serialization = 32 x slower

        }        

        __syncthreads();
    }

    //// after the outer for loop above we have 32x32 final (private and per-thread) sum's = one tile of index (blockIdx.y, blockIdx.x) of the output matrix C

    // each thread writes into the result based on it's private "sum" value
    if (row < M && col < N){
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }

}


void launch_shared_memory(
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C,
    int M, int K, int N)
{
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + threads.x - 1)/threads.x, (M + threads.y - 1)/threads.y);

    shared_memory_kernel<<<blocks, threads>>>(alpha, A, B, beta, C, M, K, N);
    CHECK_CUDA(cudaGetLastError());
}


int main(){
    std::cout << "===03_shared_memory (tiling) ===\n";
    for (int s : TEST_SIZE) {
        run_gemm_test("03_shared_memory", launch_shared_memory, s, s, s,
                      /*blockDim=*/TILE_SIZE*TILE_SIZE, /*dynamicSmemBytes=*/0,
                      (const void*)shared_memory_kernel);
    }
}

// nvcc -O3 -arch=sm_86 03_shared_memory.cu -o 03_shared_memory && ./03_shared_memory | tee 03_shared_memory.txt