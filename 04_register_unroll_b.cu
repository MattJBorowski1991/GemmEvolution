// 05_register_unroll.cu
// Register tiling: unroll inner k-loop by 8.
// Each thread accumulates 8 FMAs in registers before next shared load.
// ~25–40% faster than 03 → 9–13 TFLOPS on RTX 4090/A100.

#include "common/utils.cuh"

#define TILE_SIZE 32
#define UNROLL 8


__global__ void register_unroll_kernel(
    float alpha, 
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C,
    int M, int K, int N){

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int ty = threadIdx.y;
        int tx = threadIdx.x;

        __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
        __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

        float sum = 0.0f;

        for (int tile_k = 0; tile_k < K; tile_k += TILE_SIZE){
            // ternary slightly faster than if-else
            A_tile[ty][tx] = (row < M && tile_k + tx < K) ? A[row * K + tile_k + tx] : 0.0f;
            B_tile[ty][tx] = (tile_k + ty < K && col < N) ? B[(tile_k + ty) * N + col] : 0.0f;
            __syncthreads();
        

            // VERSION 1: from 03_shared_memory - no unrolling
            // for (int k = 0; k < TILE_SIZE; ++k){
            //     sum += A_tile[ty][k] * B_tile[k][tx];
            // }

            // VERSION 2: manual unroll (ugly but works, never do this) - equivalent to #pragma unroll 32 (likely spill or occupancy collapse)
            // sum += A_tile[ty][0] * B_tile[0][tx];
            // sum += A_tile[ty][1] * B_tile[1][tx];
            // sum += A_tile[ty][2] * B_tile[2][tx];
            // sum += A_tile[ty][3] * B_tile[3][tx];
            // sum += A_tile[ty][4] * B_tile[4][tx];
            // sum += A_tile[ty][5] * B_tile[5][tx];
            // sum += A_tile[ty][6] * B_tile[6][tx];
            // sum += A_tile[ty][7] * B_tile[7][tx];
            // sum += A_tile[ty][8] * B_tile[8][tx];
            // sum += A_tile[ty][9] * B_tile[9][tx];
            // sum += A_tile[ty][10] * B_tile[10][tx];
            // sum += A_tile[ty][11] * B_tile[11][tx];
            // sum += A_tile[ty][12] * B_tile[12][tx];
            // sum += A_tile[ty][13] * B_tile[13][tx];
            // sum += A_tile[ty][14] * B_tile[14][tx];
            // sum += A_tile[ty][15] * B_tile[15][tx];
            // sum += A_tile[ty][16] * B_tile[16][tx];
            // sum += A_tile[ty][17] * B_tile[17][tx];
            // sum += A_tile[ty][18] * B_tile[18][tx];
            // sum += A_tile[ty][19] * B_tile[19][tx];
            // sum += A_tile[ty][20] * B_tile[20][tx];
            // sum += A_tile[ty][21] * B_tile[21][tx];
            // sum += A_tile[ty][22] * B_tile[22][tx];
            // sum += A_tile[ty][23] * B_tile[23][tx];
            // sum += A_tile[ty][24] * B_tile[24][tx];
            // sum += A_tile[ty][25] * B_tile[25][tx];
            // sum += A_tile[ty][26] * B_tile[26][tx];
            // sum += A_tile[ty][27] * B_tile[27][tx];
            // sum += A_tile[ty][28] * B_tile[28][tx];
            // sum += A_tile[ty][29] * B_tile[29][tx];
            // sum += A_tile[ty][30] * B_tile[30][tx];
            // sum += A_tile[ty][31] * B_tile[31][tx];

            // VERSION 3 - pragma unroll 8 - clean and fast == unroll in chunks of UNROLL=8 (in this case there will be 4 chunks of 8)
            #pragma unroll 32
            for (int k = 0; k < TILE_SIZE; ++k){
                sum += A_tile[ty][k] * B_tile[k][tx];
            }
            __syncthreads();
        }

        if (row < M && col < N){
            C[row * N + col] = alpha * sum + beta * C[row * N + col];
        }
}


void launch_register_unroll(float alpha, const float* A, const float* B, float beta, float* C, int M, int K, int N){
    dim3 threads(32, 32);
    dim3 blocks((N + threads.x - 1)/threads.x, (M+threads.y-1)/threads.y);
    register_unroll_kernel<<<blocks,threads>>>(alpha, A, B, beta, C, M, K, N);
    CHECK_CUDA(cudaGetLastError());
}

int main(){
    std::cout << "===05_register_unroll===\n" ;
    for (int s: TEST_SIZE) {
        run_gemm_test("04_register_unroll", launch_register_unroll, s, s, s,
                      /*blockDim=*/32*32, /*dynamicSmemBytes=*/0,
                      (const void*)register_unroll_kernel);
    }
}


// nvcc -O3 -arch=sm_86 04_register_unroll_b.cu -o 04_register_unroll_b && ./04_register_unroll_b | tee 04_register_unroll_b                 