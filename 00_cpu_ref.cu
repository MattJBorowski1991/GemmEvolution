// 00_cpu_reference.cu
// Baseline: standard triple-nested CPU SGEMM (O(MNK)).
// Row-major storage, ikj loop order for cache efficiency.
// Reference for correctness and CPU performance (~7–8 GFLOPS).
// Uses all-1s matrices → exact result = K.



#include <iostream>
#include <chrono>
#include <iomanip>
#include "common/testing_sizes.hpp"
#include "common/utils.cuh"

//nvcc -O3 -arch=sm_70 00_cpu_reference.cu -o 00_cpu_reference

void cpu_sgemm(float alpha, const float* A, const float* B, float beta, float* C, int M, int K, int N) {
    for (int m = 0; m<M; ++m){
        for (int n=0; n<N; ++n){
            float sum = 0.0f;
            for (int k=0; k<K; ++k){
                sum += A[m*K + k] * B[k * N + n];
            }
            C[m * N +n] = alpha * sum + beta * C[m * N + n];
        }
    }
}

int main(){
    std::cout << "=== 00_cpu_reference ===\n";
    for (int s: TEST_SIZE){
        run_gemm_test_host("00_cpu_reference", cpu_sgemm, s, s, s);
    }
    return 0;
}

//nvcc -O3 -arch=sm_86 00_cpu_ref.cu -o 00_cpu_ref && ./00_cpu_ref | tee 00_cpu_ref.txt