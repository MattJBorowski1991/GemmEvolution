# GemmEvolution

Progressive CUDA GEMM kernels from naive to Tensor Core + double buffering, with unified benchmarking and roofline-style metrics.

## Highlights
- Kernels 01–09 showcasing key GPU optimization steps:
  - 01 Naive
  - 02 Anti-coalesced (anti-pattern)
  - 03 Shared memory tiling
  - 04 Register tiling + unrolling
  - 05 Vectorized loads (`float4`) A/B variants
  - 06 Tile-size autotuning
  - 07 WMMA (Tensor Cores): FP16 inputs → FP32 accumulates
  - 08 Double-buffered (cp.async) A/B/C variants
  - 09 cuBLAS reference (FP32 SGEMM, FP16→FP32 Tensor Ops)
- Unified metrics printed per size: time, GFLOPS, GB/s, arithmetic intensity, occupancy.
- Device header with Fpeak, Bpeak, and ridge point printed once via `run_all.sh`.
- Size sweep standardized: `{1024, 2048, 4096, 8192}`.

![Roofline plot summarizing kernel performance](roofline.png)

> Device: NVIDIA RTX A4000 (SM 86). Fpeak: 19.2 TFLOPS, Bpeak: 448 GB/s, Ridge point: 42.8 FLOPs/byte.

### Top-performing kernels (GFLOPS)

| Kernel | 1024×1024 | 2048×2048 | 4096×4096 | 8192×8192 |
| --- | --- | --- | --- | --- |
| 05_float4_vectorized (vector loads) | 1,689.8 | 1,727.2 | 1,537.6 | 1,508.5 |
| 07_tensor_wmma (Tensor Cores) | 6,114.1 | 8,260.6 | 8,518.5 | 6,020.4 |
| 08_double_buffered_c (cp.async) | **7,436.7** | **10,070.2** | **10,367.5** | **10,389.8** |

> Source: `run_all.txt` (Nov 29, 2025). Values are peak GFLOPS per kernel across the standard problem sizes.

cuBLAS FP32/FP16 baselines are still run (see `run_all.txt`) to show distance from NVIDIA-optimized kernels, but the table above spotlights only custom implementations.

Full metrics (time, bandwidth, intensity, occupancy) live in [`docs/perf_results.md`](docs/perf_results.md) and can be regenerated any time via:

```bash
python tools/perf_table.py run_all.txt docs/perf_results.md
```

## Quick Start

### Prerequisites
- CUDA Toolkit and a compatible device
- Linux/macOS shell

### Build & Run All
```bash
cd /workspace/GemmEvolution
bash run_all.sh
```
- Defaults: `ARCH=sm_86`, `NVCC=nvcc`, `CXXFLAGS=-O3`
- Override e.g. for an A100:
```bash
ARCH=sm_80 bash run_all.sh
```

### Run a Single Kernel
```bash
nvcc -O3 -arch=sm_86 03_shared_memory.cu -o 03_shared_memory
./03_shared_memory | tee 03_shared_memory.txt
```

## What Prints and Why
- Device header (once):
  - `Fpeak`: FP32 theoretical peak GFLOPS
  - `Bpeak`: theoretical memory bandwidth GB/s
  - `Ridge`: `Fpeak / Bpeak` → FLOPs/byte threshold for memory vs compute bound
- Per kernel line:
  - `ms`: elapsed time via `cudaEvent`
  - `GFLOPS`: achieved throughput
  - `GB/s`: achieved DRAM bandwidth
  - `FLOPs/byte`: arithmetic intensity (roofline x-axis)
  - `Occ`: occupancy (in `%` and `blk/SM`)

## Device Header (from `run_all.txt`)
- Printed once at the top by `run_all.sh` using `print_peaks.cu`.
- Includes:
  - `Device`: GPU name, SM version, SM count, VRAM
  - `Fpeak`: FP32 theoretical peak GFLOPS
  - `Bpeak`: theoretical memory bandwidth GB/s
  - `Ridge`: `Fpeak / Bpeak` (FLOPs/byte)

## Results Summary (excerpt from `run_all.txt`)
- Format per line: `Name MxN : ms, GFLOPS, GB/s, FLOPs/byte, Occ`
- Example selections:

```
===08_double_buffered_c (wmma runner)===
08_double_buffered_c 1024x1024 : 0.30 ms, 7182.0 GFLOPS, 42.1 GB/s, 170.67 FLOPs/byte, Occ 67% (4 blk/SM)
08_double_buffered_c 8192x8192 : 105.87 ms, 10386.0 GFLOPS, 7.6 GB/s, 1365.33 FLOPs/byte, Occ 67% (4 blk/SM)

===09_cublas (library reference)===
09_cublas_sgemm      8192x8192 : 18.78 ms, 58551.4 GFLOPS, 57.2 GB/s, 1024.00 FLOPs/byte, Occ N/A
09_cublas_fp16_tc    8192x8192 : 15.23 ms, 72211.0 GFLOPS, 52.9 GB/s, 1365.33 FLOPs/byte, Occ N/A
```

## Highlights
- Best custom kernel throughput: `08_double_buffered_c` at large sizes; compute-bound (AI >> Ridge).
- Library reference: cuBLAS achieves very high GFLOPS at 8192, useful upper bound for comparison.
- Occupancy trends: warp-level WMMA and large blocks reduce occupancy; double buffering balances resource use and utilization.
- Bandwidth scaling: achieved GB/s decreases with size while AI increases; mixes of reuse and compute dominate at large matrices.

## Roofline Interpretation
- Compute-bound if `AI > Ridge` (FLOPs/byte exceeds ridge point).
- Memory-bound if `AI < Ridge`.
- Use achieved vs peaks to reason about headroom and bottlenecks:
  - If GFLOPS ≪ Fpeak and GB/s ≪ Bpeak but `AI >> Ridge`, focus on kernel efficiency (instruction mix, register pressure, shared memory, warp scheduling).
  - If GB/s approaches Bpeak and `AI < Ridge`, optimization should target memory system (coalescing, reuse, vectorization, avoiding bank conflicts).

## Kernel Progression (High-Level)
- `01_naive`: one-thread-per-output, poor locality.
- `02_anti_coalesced`: deliberately breaks coalescing (for contrast).
- `03_shared_memory`: tile A/B in shared memory; coalesced loads; reuse.
- `04_register_unroll_{a,b}`: accumulate in registers; unroll loops; fewer instructions.
- `05_float4_vectorized_{a,b}`: vectorized global loads/stores; improved bandwidth utilization.
- `06_autotune_tiles`: sweep tile sizes, pick best by timing.
- `07_tensor_core_wmma`: warp-level matrix multiply using Tensor Cores, FP16→FP32.
- `08_double_buffered_{a,b,c}`: pipeline global→shared copies with `cp.async` while computing.
- `09_cublas`: FP32 SGEMM and FP16→FP32 reference via cuBLAS, useful for sanity/perf comparison.

## Files
- `common/utils.cuh`: shared runners (allocation, timing, metrics formatting, occupancy).
- `common/metrics.cuh`: bytes moved, bandwidth, arithmetic intensity, occupancy helpers, Fpeak/Bpeak.
- `common/testing_sizes.hpp`: shared sizes `{1024, 2048, 4096, 8192}`.
- `run_all.sh`: builds 01–09, prints device peaks, runs kernels, logs to `run_all.txt` and per-file `*.txt`.
- `print_peaks.cu`: small helper program used by `run_all.sh` to emit device Fpeak, Bpeak, ridge.

## Correctness
- The runners initialize inputs and can compare against CPU or cuBLAS if you enable checks. For portfolio brevity, correctness checks are omitted by default; you can add small-size comparisons using `00_cpu_ref` or cuBLAS.

## Tips for Reviewers (What This Demonstrates)
- Global vs shared memory access patterns and coalescing.
- Register tiling, loop unrolling, and vectorized memory ops.
- Occupancy trade-offs with block sizes, registers, and shared memory.
- Warp-level WMMA/Tensor Core usage.
- Double buffering with `cp.async` (Ampere+).
- Clean measurement discipline (events), consistent metrics, and roofline framing.

## Common Tweaks
- Change GPU architecture: `ARCH=sm_XX` in `run_all.sh`.
- Adjust sizes: edit `common/testing_sizes.hpp`.
- Enable warmup iterations: add a pre-run inside runners for more stable small-size timings.
- Add percent-of-peak fields: compute `GFLOPS% = achieved / Fpeak`, `GB/s% = achieved / Bpeak`.

## License
- This repository is intended as a learning and portfolio artifact. No external copyrighted code is included.
