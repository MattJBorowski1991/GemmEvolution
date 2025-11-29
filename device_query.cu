#include <cstdio>
#include <cuda_runtime.h>

int main(){
    int devCount=0; cudaGetDeviceCount(&devCount);
    if(devCount<=0){ printf("No CUDA devices found\n"); return 0; }
    int dev=0; cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    printf("Device %d: %s\n", dev, prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SM Count (multiProcessorCount): %d\n", prop.multiProcessorCount);
    printf("Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads Dim: [%d, %d, %d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max Grid Size: [%d, %d, %d]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("Shared Mem Per Block: %zu bytes\n", (size_t)prop.sharedMemPerBlock);
    printf("Shared Mem Per Multiprocessor: %zu bytes\n", (size_t)prop.sharedMemPerMultiprocessor);
    printf("Registers Per Block: %d\n", prop.regsPerBlock);
    printf("Regs Per Multiprocessor: %d\n", prop.regsPerMultiprocessor);
    printf("Max Blocks Per Multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("Max Threads Per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Memory Clock Rate: %d kHz\n", prop.memoryClockRate);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("Total Global Memory: %zu bytes\n", (size_t)prop.totalGlobalMem);
    printf("L2 Cache Size: %d bytes\n", prop.l2CacheSize);
    printf("Concurrent Kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
    printf("Cooperative Launch: %s\n", prop.cooperativeLaunch ? "Yes" : "No");
    printf("Max Shared Mem Optin Per Block: %zu bytes\n", (size_t)prop.sharedMemPerBlockOptin);
    return 0;
}
