#!/usr/bin/env python3
"""
Generate a roofline chart from run_all.txt.
- Parses lines: "Name MxN : ms, GFLOPS, GB/s, FLOPs/byte, Occ ..."
- Requires a header printed by run_all.sh containing Fpeak, Bpeak, Ridge.
- Outputs roofline.png in the current directory.
"""
import re
import os
import sys
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

RUN_FILE = sys.argv[1] if len(sys.argv) > 1 else 'run_all.txt'
OUT_FILE = sys.argv[2] if len(sys.argv) > 2 else 'roofline.png'

fpeak = None
bpeak = None
ridge = None
points = []  # (name, size, ai, gflops)

name_re = re.compile(r"^([A-Za-z0-9_]+?)(?=\s*\d+x)\s*(\d+)x(\d+)\s*:\s*(.+)$")
nums_re = re.compile(r"([0-9]+\.?[0-9]*)")

with open(RUN_FILE, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Parse peaks
        if line.startswith('Fpeak:'):
            fpeak = float(line.split(':',1)[1].strip().split()[0])
            continue
        if line.startswith('Bpeak:'):
            bpeak = float(line.split(':',1)[1].strip().split()[0])
            continue
        if line.startswith('Ridge:'):
            ridge = float(line.split(':',1)[1].strip().split()[0])
            continue
        m = name_re.match(line)
        if not m:
            continue
        name = m.group(1)
        if name == '00_cpu_ref':  # Skip CPU reference
            continue
        size = m.group(2)
        tail = m.group(4)
        # Expect sequence: ms, GFLOPS, GB/s, FLOPs/byte, Occ
        nums = nums_re.findall(tail)
        if len(nums) < 4:
            continue
        ms = float(nums[0])
        gflops = float(nums[1])
        gbps = float(nums[2])
        ai = float(nums[3])
        points.append((name, size, ai, gflops))

if fpeak is None or bpeak is None:
    print('Error: Fpeak/Bpeak not found in run_all.txt header.')
    sys.exit(1)
if ridge is None:
    ridge = fpeak / bpeak

# Build roofline curves
xs = []
ys = []
max_ai = max([p[2] for p in points]) if points else 100.0
x_min = 0.1
x_max = max_ai * 1.2
num_samples = 200
for i in range(num_samples):
    x = x_min + (x_max - x_min) * i / (num_samples - 1)
    y = min(fpeak, bpeak * x)
    xs.append(x)
    ys.append(y)

plt.figure(figsize=(10, 6))
plt.plot(xs, ys, label='Roofline', color='black')
plt.axvline(ridge, linestyle='--', color='gray', alpha=0.6, label=f'Ridge = {ridge:.2f}')

# Get a list of unique kernels and sort them to ensure consistent ordering
kernels = sorted(list(set([p[0] for p in points])))

# Create a custom color palette with more distinct colors
# Using a combination of tableau colors and CSS4 colors for better distinction
colors = [
    'tab:blue',    # 01_naive
    'tab:orange',  # 02_global_mem_coalescing
    'tab:green',   # 03_shared_mem
    'tab:red',     # 04_1d_reg_tiling
    'tab:purple',  # 05_2d_reg_tiling
    'tab:brown',   # 06_loop_unrolling
    'tab:pink',    # 07_prefetching
    'tab:olive',   # 08_smem_padding
    'tab:cyan',    # 09_cublas_sgemm (changed from default to be more distinct)
    'darkviolet',  # Additional distinct colors if needed
    'darkorange',
    'darkgreen',
    'darkred'
]


def legend_label(kernel_name: str) -> str:
    if kernel_name.startswith('05_float4_vectorized'):
        return '05_float4_vectorized'
    if kernel_name.startswith('08_double_buffered'):
        return '08_double_buffered'
    if kernel_name.startswith('06_'):
        return '06_tiles'
    return kernel_name


color_map = {}


def pick_color(label: str) -> str:
    if label not in color_map:
        color_map[label] = colors[len(color_map) % len(colors)]
    return color_map[label]

# Create a scatter plot for each kernel
handles = []
labels = []

# First pass: collect all unique kernel names and their styles
for i, kernel in enumerate(kernels):
    kernel_points = [p for p in points if p[0] == kernel]
    if not kernel_points:
        continue

    x = [p[2] for p in kernel_points]
    y = [p[3] for p in kernel_points]
    legend_name = legend_label(kernel)

    if kernel == '01_naive':
        sc = plt.scatter(x, y, facecolors='none', edgecolors='black', linewidth=1.5, s=80)
    elif kernel == '09_cublas_fp16_tc':
        edge_color = pick_color('08_double_buffered')
        sc = plt.scatter(x, y, facecolors='none', edgecolors=edge_color, linewidth=1.5, s=80)
    elif kernel.startswith('05_float4_vectorized'):
        sc = plt.scatter(x, y, color='darkviolet', s=50)
    elif kernel.startswith('08_double_buffered'):
        sc = plt.scatter(x, y, color=pick_color(legend_name), s=50)
    elif kernel.startswith('06_'):
        sc = plt.scatter(x, y, facecolors='none', edgecolors='red', linewidth=1.5, s=70)
    elif kernel == '09_cublas_sgemm':
        sc = plt.scatter(x, y, facecolors='none', edgecolors='darkviolet', linewidth=1.5, s=80)
    else:
        sc = plt.scatter(x, y, color=pick_color(legend_name), s=50)

    # Only add to legend if not already added
    if legend_name not in labels:
        handles.append(sc)
        labels.append(legend_name)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Arithmetic Intensity (FLOPs/byte)')
plt.ylabel('GFLOPS')
plt.title('Roofline: Custom Kernels vs Device Peaks')
plt.grid(True, which='both', ls=':', alpha=0.4)

# Position legend outside the plot to the right
plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout()
plt.savefig(OUT_FILE, bbox_inches='tight', dpi=300)
print(f"Roofline plot saved to {OUT_FILE}")