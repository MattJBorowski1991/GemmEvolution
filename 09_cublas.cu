// 09_cublas.cu
// Integrate cuBLAS reference GEMMs using existing runners:
// - FP32 SGEMM via run_gemm_test (float path)
// - FP16->FP32 Tensor Core GEMM via run_gemm_test_wmma (half inputs, float output)
// Occupancy printed as N/A (no device kernel symbol provided).

#include "common/utils.cuh"
#include <cublas_v2.h>

#define CHECK_CUBLAS(call) { \
    cublasStatus_t st = call; \
    if (st != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error code " << st << " at line " << __LINE__ << "\n"; \
        exit(1); \
    } \
}

// Lazy-initialized handle
static cublasHandle_t get_handle(){
    static cublasHandle_t handle = nullptr;
    if (!handle){
        CHECK_CUBLAS(cublasCreate(&handle));
        cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    }
    return handle;
}

// Signature matches run_gemm_test expectations for float path
void launch_cublas_sgemm(float alpha, const float* A, const float* B, float beta, float* C, int M, int K, int N){
    // Row-major GEMM: we stored A,B,C row-major; cublas assumes column-major.
    // Compute C = A*B (row-major) by calling column-major equivalent: C^T = B^T * A^T.
    cublasHandle_t h = get_handle();
    CHECK_CUBLAS(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                             B, N, A, K, &beta, C, N));
}

// Signature matches run_gemm_test_wmma (half inputs, float output)
void launch_cublas_fp16_tc(float alpha, const __half* A, const __half* B, float beta, float* C, int M, int K, int N){
    cublasHandle_t h = get_handle();
    CHECK_CUBLAS(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K,
                              &alpha,
                              B, CUDA_R_16F, N,
                              A, CUDA_R_16F, K,
                              &beta,
                              C, CUDA_R_32F, N,
                              CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

int main(){
    std::cout << "===09_cublas (library reference)===\n";
    for (int s : TEST_SIZE){
        // FP32 path (no occupancy kernel symbol)
        run_gemm_test("09_cublas_sgemm", launch_cublas_sgemm, s, s, s,
                      /*blockDim=*/0, /*dynamicSmemBytes=*/0, /*deviceKernel=*/nullptr);
        // FP16->FP32 Tensor Core path
        run_gemm_test_wmma("09_cublas_fp16_tc", launch_cublas_fp16_tc, s, s, s,
                           /*blockDim=*/0, /*dynamicSmemBytes=*/0, /*deviceKernel=*/nullptr,
                           1.0f, 0.0f);
    }
    return 0;
}

//nvcc -O3 -arch=sm_86 09_cublas.cu -lcublas -o 09_cublas ./09_cublas | tee 09_cublas.txt
