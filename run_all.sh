#!/usr/bin/env bash
set -euo pipefail

# Compile and run all CUDA files from 01 to 09 (excluding 00)
# Outputs are tee'd to matching .txt files.

ARCH=${ARCH:-sm_86}
NVCC=${NVCC:-nvcc}
CXXFLAGS=${CXXFLAGS:-"-O3"}

cd "$(dirname "$0")"

# Find target .cu files whose basename starts with 01..08 plus 09 (cuBLAS)
mapfile -t files < <(find . -maxdepth 1 -type f -regex './0[1-9].*\.cu' | sort)

# Prepare aggregated output file
:> run_all.txt
echo "Run started: $(date)" | tee -a run_all.txt

# Print device peaks once at the top (Fpeak, Bpeak, ridge)
echo "==> Computing device peaks" | tee -a run_all.txt
${NVCC} ${CXXFLAGS} -arch=${ARCH} print_peaks.cu -o print_peaks
./print_peaks | tee -a run_all.txt

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No target .cu files found (01..08)." >&2
  exit 1
fi

for cu in "${files[@]}"; do
  base="${cu##./}"
  exe="${base%.cu}"
  txt="${exe}.txt"
  {
    echo "==> Building ${base}"
    # Link cuBLAS only for 09_cublas
    if [[ "${exe}" == "09_cublas" ]]; then
      ${NVCC} ${CXXFLAGS} -arch=${ARCH} "${cu}" -lcublas -o "${exe}"
    else
      ${NVCC} ${CXXFLAGS} -arch=${ARCH} "${cu}" -o "${exe}"
    fi
    echo "==> Running ./${exe}"
    "./${exe}"
    echo "-- Completed ${exe}"
    echo
  } | tee -a run_all.txt | tee -a "${txt}"
done

echo "All runs completed." | tee -a run_all.txt
echo "Run finished: $(date)" | tee -a run_all.txt