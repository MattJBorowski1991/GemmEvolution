// 07_tensor_wmma.cu
// Minimal Tensor Core GEMM (WMMA) – FP16 input, FP32 output
// 900–1,200+ TFLOPS on modern GPUs


#include "common/utils.cuh"
#include <mma.h>        // Tensor Core header - enables Tensor Cores
using namespace nvcuda;  // needed for wmma::

__global__ void wmma_kernel(
    float alpha,
    const half* __restrict__ A,         //FP16 for matrix A - saves bandwiths, feeds TCores, as FP32 not supported
    const half* __restrict__ B,         //FP16 for matrix B
    float beta,
    float* __restrict__ C,              //FP32 for output matrix
    int M, int K, int N
)
{

    // shape of Tensor Core tile = 16x16x16
    constexpr int WMMA_M = 16;          //compile-time constant = value known at compile time = required by WMMA
    constexpr int WMMA_K = 16;
    constexpr int WMMA_N = 16;


    // One warp (32 threads) computes ONE 16×16 output tile
    // → 32 threads cooperate to load & multiply 16×16×16 = 4096 ops
    // → 3. How? 32 threads = exactly enough to hold all data + control

    //We do not use threadIdx at all as WMMA is Warp-level, not thread-level. 
    //32 threads in the warp cooperate automatically, hence you don't need thread ID

    const int tile_row = blockIdx.y * WMMA_M;
    const int tile_col = blockIdx.x * WMMA_N;

    if (tile_row >= M || tile_col >= N) return;

    // Fragments = special Tensor Core registers
    // They live in the warp's registers, not normal registers
    // the sequence has to be w, n, k in the fragments' declarations below
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);      // C = 0

    //Loop over K dimension in 16-wide chunks
    for (int k=0; k < K; k += WMMA_K){
        int a_col = k;
        int b_row = k;
        
        //Load 16x16 pieces of A and B directly from Global Memory
        const half* a_ptr = A + tile_row * K + a_col;
        const half* b_ptr = B + b_row * N + tile_col;

        if (a_col + WMMA_K <= K){           // Bounds check
            wmma::load_matrix_sync(a_frag, a_ptr, K);       // 32 threads load 256 halfs - WMMA automatically loads the full 16x16 tile starting from the tile's upper left corner
            wmma::load_matrix_sync(b_frag, b_ptr, N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // 4096 MMAs in ONE instruction!!!
        }
    }
    //After the for loop above there is one 16x16 tile per block calculated. It is accumulated in c_frag


    // (pointer to first element of large matrix C) + (1D index of the top-left corner of our 16x16 tile) = pointer to C[tile_row][tile_col]
    // the first '+' here is pointer arithmetic == move forward by (tile_row*N+tile_col)-many floats
    // C destination pointer = memory address where we will write our final 16x16 result
    float* c_dst = C + (tile_row * N + tile_col);

    //load original C, calculate GEMM
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_old;
    wmma::load_matrix_sync(c_old, c_dst, N, wmma::mem_row_major);
    for (int i = 0; i < c_old.num_elements; ++i)
        c_frag.x[i] = alpha * c_frag.x[i] + beta * c_old.x[i];
    
    //take the finished 16x16 from c_frag and write it directly into global memory starting at c_dst - pointing to C[tile_row][tile_col]
    // this is done for every block
    wmma::store_matrix_sync(c_dst, c_frag, N, wmma::mem_row_major);
}


// Launcher matching run_gemm_test_wmma signature
void launch_wmma(float alpha, const half* A, const half* B, float beta, float* C, int M, int K, int N){
    dim3 block(32); // one warp per 16x16 tile. we don't do block(16,16), as WMMA is Warp-level and one WMMA tile = 1 warp = 32 threads = 1 block
    dim3 grid((N+15)/16, (M+15)/16);
    wmma_kernel<<<grid, block>>>(alpha, A, B, beta, C, M, K, N);
    CHECK_CUDA(cudaGetLastError());
}

int main(){
    std::cout << "===07_tensor_wmma===\n";
    for (int s : TEST_SIZE){
        run_gemm_test_wmma("07_tensor_wmma", launch_wmma, s, s, s,
                   /*blockDim=*/32, /*dynamicSmemBytes=*/0,
                   (const void*)wmma_kernel,
                   1.0f, 0.0f);
    }
    return 0;
}


// nvcc -O3 -arch=sm_86 07_tensor_core_wmma.cu -o 07_tensor_core_wmma && ./07_tensor_core_wmma | tee 07_tensor_core_wmma.txt