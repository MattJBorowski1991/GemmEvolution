#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "testing_sizes.hpp"
#include "metrics.cuh"


#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

// --- Occupancy (computed in metrics.cuh) ---
// Runners accept device kernel symbol plus block/shared config and
// compute occupancy via metrics::compute_occupancy.

inline void run_gemm_test(
    const char* name, // e.g. "01_naive"
    void (*kernel)(float, const float*, const float*, float, float*, int, int, int), int M, int K, int N,
    int blockDim = 0, std::size_t dynamicSmemBytes = 0,
    const void* deviceKernel = nullptr
)
{
    float *h_A = new float[M*K];
    float *h_B = new float[K*N];
    float *h_C = new float[M*N];

    float *d_A, *d_B, *d_C;

    std::fill_n(h_A, M*K, 1.0f);
    std::fill_n(h_B, K*N, 1.0f);
    std::fill_n(h_C, M*N, 0.0f);

    CHECK_CUDA(cudaMalloc(&d_A, M*K*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M*N*sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, M*N*sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    // launcher should accept (alpha, A, B, beta, C, M, K, N)
    kernel(1.0f, d_A, d_B, 0.0f, d_C, M, K, N);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

float ms = 0.0f;

    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = flops * 1e-9 / (ms / 1000.0);
    std::size_t bytes = metrics::bytes_moved_gemm<float>(M, N, K, /*readC=*/true, /*writeC=*/true);
    double gbps = metrics::effective_bandwidth_GBps(ms, bytes);
    double ai = metrics::arithmetic_intensity(flops, (double)bytes);

    int activeBlocks = -1; double occ = -1.0;
    if (deviceKernel && blockDim > 0) {
        auto p = metrics::compute_occupancy(deviceKernel, blockDim, dynamicSmemBytes);
        activeBlocks = p.first; occ = p.second;
    }

    std::cout   << std::left << std::setw(20) << name << M << "x" << N << " : " 
                << std::fixed << std::setprecision(2) << ms << " ms, " 
                << std::setprecision(1) << gflops << " GFLOPS, "
                << std::setprecision(1) << gbps << " GB/s, "
                << std::setprecision(2) << ai << " FLOPs/byte"
                << ((activeBlocks >= 0 && occ >= 0.0) ? 
                    (std::ostringstream() << ", Occ " << std::setprecision(0) << std::fixed << (occ*100.0) << "% (" << activeBlocks << " blk/SM)").str() : ", Occ N/A")
                << "\n";

CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));
CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
delete[] h_A; delete[] h_B; delete[] h_C;
}


// CPU version overload
inline void run_gemm_test_host(
    const char* name,
    void (*kernel)(float, const float*, const float*, float, float*, int, int, int),
    int M, int K, int N)
{
    float *A = new float[M*K];
    float *B = new float[K*N];
    float *C = new float[M*N];
    std::fill_n(A, M*K, 1.0f);
    std::fill_n(B, K*N, 1.0f);
    std::fill_n(C, M*N, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();
    // host kernel expects (alpha, A, B, beta, C, M, K, N)
    kernel(1.0f, A, B, 0.0f, C, M, K, N);
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    double gflops = flops * 1e-9 / (ms / 1000.0);
    std::size_t bytes = metrics::bytes_moved_gemm<float>(M, N, K, /*readC=*/true, /*writeC=*/true);
    double gbps = metrics::effective_bandwidth_GBps(ms, bytes);
    double ai = metrics::arithmetic_intensity(flops, static_cast<double>(bytes));

    std::cout << std::left << std::setw(20) << name
              << M << "x" << N << " : "
              << std::fixed << std::setprecision(2) << ms << " ms, "
              << std::setprecision(1) << gflops << " GFLOPS, "
              << std::setprecision(1) << gbps << " GB/s, "
              << std::setprecision(2) << ai << " FLOPs/byte, Occ N/A\n";

    delete[] A; delete[] B; delete[] C;
}

// GPU version with B transposed
inline void run_gemm_test_transposed(
    const char* name,
    void (*kernel)(float, const float*, const float*, float, float*, int, int, int),
    int M, int K, int N,
    int blockDim = 0, std::size_t dynamicSmemBytes = 0,
    const void* deviceKernel = nullptr)
{
    float *h_A = new float[M*K];
    float *h_B = new float[K*N];
    float *h_C = new float[M*N];

    std::fill_n(h_A, M*K, 1.0f);
    std::fill_n(h_B, K*N, 1.0f);
    std::fill_n(h_C, M*N, 0.0f);

    // Transpose B on host
    float *h_B_t = new float[N*K];
    for (int n = 0; n < N; ++n){
        for (int k = 0; k < K; ++k){
            h_B_t[n * K + k] = h_B[k * N + n];
        }
    }

    float *d_A, *d_B_t, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M*K*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B_t, N*K*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M*N*sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_t, h_B_t, N*K*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, M*N*sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    kernel(1.0f, d_A, d_B_t, 0.0f, d_C, M, K, N);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = flops * 1e-9 / (ms / 1000.0);
    std::size_t bytes = metrics::bytes_moved_gemm<float>(M, N, K, /*readC=*/true, /*writeC=*/true);
    double gbps = metrics::effective_bandwidth_GBps(ms, bytes);
    double ai = metrics::arithmetic_intensity(flops, (double)bytes);

    int activeBlocks = -1; double occ = -1.0;
    if (deviceKernel && blockDim > 0) {
        auto p = metrics::compute_occupancy(deviceKernel, blockDim, dynamicSmemBytes);
        activeBlocks = p.first; occ = p.second;
    }

    std::cout << std::left << std::setw(20) << name << M << "x" << N << " : " 
              << std::fixed << std::setprecision(2) << ms << " ms, " 
              << std::setprecision(1) << gflops << " GFLOPS, "
              << std::setprecision(1) << gbps << " GB/s, "
              << std::setprecision(2) << ai << " FLOPs/byte"
              << ((activeBlocks >= 0 && occ >= 0.0) ? 
                  (std::ostringstream() << ", Occ " << std::setprecision(0) << std::fixed << (occ*100.0) << "% (" << activeBlocks << " blk/SM)").str() : ", Occ N/A")
              << "\n";

    CHECK_CUDA(cudaFree(d_A)); 
    CHECK_CUDA(cudaFree(d_B_t)); 
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start)); 
    CHECK_CUDA(cudaEventDestroy(stop));
    delete[] h_A; 
    delete[] h_B; 
    delete[] h_B_t;
    delete[] h_C;
}

// WMMA (Tensor Core) runner: A,B in half, C in float
inline void run_gemm_test_wmma(
    const char* name,
    void (*kernel)(float, const __half*, const __half*, float, float*, int, int, int),
    int M, int K, int N,
    int blockDim = 0, std::size_t dynamicSmemBytes = 0,
    const void* deviceKernel = nullptr,
    float alpha = 1.0f,
    float beta  = 0.0f)
{
    // Allocate device buffers
    __half *d_A = nullptr; __half *d_B = nullptr; float *d_C = nullptr;
    CHECK_CUDA(cudaMalloc(&d_A, M*K*sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_B, K*N*sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C, M*N*sizeof(float)));

    // Host init to 1.0f in half, 0.0f in C
    std::vector<__half> h_A(M*K), h_B(K*N);
    for (auto &x : h_A) x = __float2half(1.0f);
    for (auto &x : h_B) x = __float2half(1.0f);
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M*K*sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K*N*sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, M*N*sizeof(float)));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    kernel(alpha, d_A, d_B, beta, d_C, M, K, N);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f; CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = flops * 1e-9 / (ms / 1000.0);
    std::size_t bytes = (std::size_t)M*K*sizeof(__half) + (std::size_t)K*N*sizeof(__half)
                      + (std::size_t)M*N*sizeof(float)  /* write C */
                      + (std::size_t)M*N*sizeof(float); /* read C */
    double gbps = metrics::effective_bandwidth_GBps(ms, bytes);
    double ai = metrics::arithmetic_intensity(flops, (double)bytes);

    int activeBlocks = -1; double occ = -1.0;
    if (deviceKernel && blockDim > 0) {
        auto p = metrics::compute_occupancy(deviceKernel, blockDim, dynamicSmemBytes);
        activeBlocks = p.first; occ = p.second;
    }

    std::cout << std::left << std::setw(20) << name << M << "x" << N << " : "
              << std::fixed << std::setprecision(2) << ms << " ms, "
              << std::setprecision(1) << gflops << " GFLOPS, "
              << std::setprecision(1) << gbps << " GB/s, "
              << std::setprecision(2) << ai << " FLOPs/byte"
              << ((activeBlocks >= 0 && occ >= 0.0) ? 
                  (std::ostringstream() << ", Occ " << std::setprecision(0) << std::fixed << (occ*100.0) << "% (" << activeBlocks << " blk/SM)").str() : ", Occ N/A")
              << "\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));
}

// GPU runner: pre-transpose B on host (outside timing), time only kernel on B_t (N x K)
inline void run_once_vectorized_b(
    const char* name,
    void (*kernel)(float, const float*, const float*, float, float*, int, int, int),
    int M, int K, int N,
    int blockDim = 0, std::size_t dynamicSmemBytes = 0,
    const void* deviceKernel = nullptr)
{
    float *h_A = new float[M*K];
    float *h_B = new float[K*N];
    float *h_C = new float[M*N];

    std::fill_n(h_A, M*K, 1.0f);
    std::fill_n(h_B, K*N, 1.0f);
    std::fill_n(h_C, M*N, 0.0f);

    // Host pre-transpose B → B_t (outside timed region)
    float *h_B_t = new float[N*K];
    for (int n = 0; n < N; ++n){
        for (int k = 0; k < K; ++k){
            h_B_t[n * K + k] = h_B[k * N + n];
        }
    }

    float *d_A=nullptr, *d_B_t=nullptr, *d_C=nullptr;
    CHECK_CUDA(cudaMalloc(&d_A, M*K*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B_t, N*K*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M*N*sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_t, h_B_t, N*K*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, M*N*sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    // Kernel expects B_t layout
    kernel(1.0f, d_A, d_B_t, 0.0f, d_C, M, K, N);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = flops * 1e-9 / (ms / 1000.0);
    std::size_t bytes = metrics::bytes_moved_gemm<float>(M, N, K, /*readC=*/true, /*writeC=*/true);
    double gbps = metrics::effective_bandwidth_GBps(ms, bytes);
    double ai = metrics::arithmetic_intensity(flops, (double)bytes);

    int activeBlocks = -1; double occ = -1.0;
    if (deviceKernel && blockDim > 0) {
        auto p = metrics::compute_occupancy(deviceKernel, blockDim, dynamicSmemBytes);
        activeBlocks = p.first; occ = p.second;
    }

    std::cout << std::left << std::setw(20) << name << M << "x" << N << " : "
              << std::fixed << std::setprecision(2) << ms << " ms, "
              << std::setprecision(1) << gflops << " GFLOPS, "
              << std::setprecision(1) << gbps << " GB/s, "
              << std::setprecision(2) << ai << " FLOPs/byte"
              << ((activeBlocks >= 0 && occ >= 0.0) ? 
                  (std::ostringstream() << ", Occ " << std::setprecision(0) << std::fixed << (occ*100.0) << "% (" << activeBlocks << " blk/SM)").str() : ", Occ N/A")
              << "\n";

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B_t));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    delete[] h_A;
    delete[] h_B;
    delete[] h_B_t;
    delete[] h_C;
}

