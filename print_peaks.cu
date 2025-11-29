#include <iostream>
#include <cuda_runtime.h>
#include "common/metrics.cuh"

int main(){
    int dev=0; cudaGetDevice(&dev);
    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, dev);
    double fpeak = metrics::theoretical_peak_fp32_gflops(dev);
    double bpeak = metrics::theoretical_peak_mem_bandwidth_GBps(dev);
    double ridge = (bpeak > 0.0) ? (fpeak / bpeak) : 0.0; // FLOPs/byte
    std::cout.setf(std::ios::fixed); std::cout.precision(1);
    std::cout << "Device: " << prop.name << " (SM " << prop.major << prop.minor << ")\n";
    std::cout << "Fpeak: " << fpeak << " GFLOPS\n";
    std::cout << "Bpeak: " << bpeak << " GB/s\n";
    std::cout.precision(2);
    std::cout << "Ridge: " << ridge << " FLOPs/byte\n";
    return 0;
}
