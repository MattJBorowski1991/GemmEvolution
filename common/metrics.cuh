#pragma once
#include <cstddef>
#include <cuda_runtime.h>

namespace metrics {

// Minimal helpers for effective memory bandwidth (GB/s)
//
// Assumptions (SGEMM-style):
// - Reads A[M*K] and B[K*N]
// - Optionally reads C[M*N] when beta != 0 (set readC=true)
// - Writes C[M*N] by default (set writeC=false if not writing)
//
// Keep this small and header-only so any .cu can include it.

// Compute total bytes moved for GEMM-like access (templated on element type T)
template <typename T>
inline std::size_t bytes_moved_gemm(std::size_t M,
                                    std::size_t N,
                                    std::size_t K,
                                    bool readC  = false,
                                    bool writeC = true) {
    const std::size_t a = M * K;
    const std::size_t b = K * N;
    const std::size_t c = M * N;
    const std::size_t reads  = a + b + (readC ? c : 0);
    const std::size_t writes = writeC ? c : 0;
    return (reads + writes) * sizeof(T);
}

// Convert elapsed milliseconds and total bytes moved into GB/s
inline double effective_bandwidth_GBps(double elapsed_ms, std::size_t bytes_moved) {
    const double seconds = elapsed_ms / 1000.0;
    return (seconds > 0.0) ? (static_cast<double>(bytes_moved) / seconds) / 1e9 : 0.0;
}

// Arithmetic intensity: FLOPs per byte moved
inline double arithmetic_intensity(double flops, std::size_t bytes_moved) {
    return (bytes_moved > 0) ? flops / static_cast<double>(bytes_moved) : 0.0;
}

// --- Occupancy helpers (Nsight-free) ---
// Returns max warps per SM for the active device.
inline int maxWarpsPerSM(){
    cudaDeviceProp prop{}; int dev = 0; cudaGetDevice(&dev); cudaGetDeviceProperties(&prop, dev);
    // Each SM has prop.maxThreadsPerMultiProcessor threads capacity; warps are 32 threads.
    return prop.maxThreadsPerMultiProcessor / 32;
}

// Compute theoretical occupancy given active blocks per SM and blockDim.
// occupancy = activeWarpsPerSM / maxWarpsPerSM, clamped to [0,1].
inline double occupancyFromActiveBlocks(int activeBlocksPerSM, int blockDim){
    int warpsPerBlock = (blockDim + 31) / 32;
    int activeWarps = activeBlocksPerSM * warpsPerBlock;
    int maxWarps = maxWarpsPerSM();
    if (maxWarps <= 0) return 0.0;
    double occ = static_cast<double>(activeWarps) / static_cast<double>(maxWarps);
    if (occ < 0.0) occ = 0.0; if (occ > 1.0) occ = 1.0;
    return occ;
}

// Compute occupancy using CUDA API given a device kernel symbol.
// Returns pair (activeBlocksPerSM, occupancy in [0,1]).
inline std::pair<int,double> compute_occupancy(const void* deviceKernel,
                                               int blockSize,
                                               std::size_t dynamicSmemBytes){
    int activeBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocksPerSM,
        deviceKernel,
        blockSize,
        static_cast<size_t>(dynamicSmemBytes));
    double occ = occupancyFromActiveBlocks(activeBlocksPerSM, blockSize);
    return {activeBlocksPerSM, occ};
}

// --- Theoretical peak device metrics ---
// Helper to map compute capability to FP32 cores per SM.
// Note: Ampere GA100 (sm_80) has 64 FP32 cores/SM; GA10x (sm_86) has 128.
// Values sourced from NVIDIA public architecture specifications.
inline int fp32_cores_per_sm(int major, int minor){
    int cc = major * 10 + minor;
    switch (cc){
        case 50: // Maxwell
        case 52:
        case 53:
            return 128;
        case 60: // Pascal GP100
            return 64;
        case 61: // Pascal consumer
        case 62:
            return 128;
        case 70: // Volta
        case 72: // Xavier (Volta derivative)
            return 64;
        case 75: // Turing
            return 64;
        case 80: // Ampere A100
            return 64;
        case 86: // Ampere GA10x (e.g. RTX 30 series)
            return 128;
        case 89: // Ada Lovelace (e.g. RTX 40, SM89: 128 FP32 per SM)
            return 128;
        case 90: // Hopper (SM90: 64 FP32 per SM)
            return 64;
        default:
            // Fallback: assume 64; conservative for newer architectures.
            return 64;
    }
}

// Theoretical FP32 peak in GFLOPS (FMA counts as 2 FLOPs).
inline double theoretical_peak_fp32_gflops(int device = -1){
    if (device < 0) cudaGetDevice(&device);
    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, device);
    const int coresPerSM = fp32_cores_per_sm(prop.major, prop.minor);
    // clockRate is in kHz; convert to Hz.
    const double smClockHz = static_cast<double>(prop.clockRate) * 1000.0;
    // 2 FLOPs per FP32 core per cycle (FMA).
    const double flopsPerSecond = static_cast<double>(prop.multiProcessorCount) * coresPerSM * 2.0 * smClockHz;
    return flopsPerSecond / 1e9; // GFLOPS
}

// Theoretical memory bandwidth in GB/s.
// Assumes double data rate (GDDR / HBM) => dramRateMultiplier=2.
inline double theoretical_peak_mem_bandwidth_GBps(int device = -1, int dramRateMultiplier = 2){
    if (device < 0) cudaGetDevice(&device);
    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, device);
    // memoryClockRate is in kHz; convert to Hz.
    if (prop.memoryClockRate == 0 || prop.memoryBusWidth == 0) return 0.0; // Fallback if unavailable.
    const double memClockHz = static_cast<double>(prop.memoryClockRate) * 1000.0;
    const double busBytes = static_cast<double>(prop.memoryBusWidth) / 8.0; // bits -> bytes
    const double bytesPerSecond = memClockHz * busBytes * dramRateMultiplier;
    return bytesPerSecond / 1e9; // GB/s
}

// Percent of peak helper (returns 0 if peak==0).
inline double percent_of_peak(double achieved, double peak){
    return (peak > 0.0) ? (achieved / peak) * 100.0 : 0.0;
}


} // namespace metrics
