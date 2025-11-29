#!/usr/bin/env python3
"""Generate a Markdown performance table from run_all.txt outputs."""
from __future__ import annotations

import re
import sys
from pathlib import Path

RUN_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("run_all.txt")
OUT_FILE = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("docs/perf_results.md")

run_started = None
run_finished = None
device = None
fpeak = None
bpeak = None
ridge = None
rows = []

name_re = re.compile(r"^([A-Za-z0-9_]+?)(?=\s*\d+x)\s*(\d+)x(\d+)\s*:\s*(.+)$")
metrics_re = re.compile(
    r"([0-9]+\.?[0-9]*)\s+ms,\s+"
    r"([0-9]+\.?[0-9]*)\s+GFLOPS,\s+"
    r"([0-9]+\.?[0-9]*)\s+GB/s,\s+"
    r"([0-9]+\.?[0-9]*)\s+FLOPs/byte,\s+Occ\s+(.+)"
)

with RUN_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Run started:"):
            run_started = line.split(":", 1)[1].strip()
            continue
        if line.startswith("Run finished:"):
            run_finished = line.split(":", 1)[1].strip()
            continue
        if line.startswith("Device:"):
            device = line.split(":", 1)[1].strip()
            continue
        if line.startswith("Fpeak:"):
            fpeak = line.split(":", 1)[1].strip()
            continue
        if line.startswith("Bpeak:"):
            bpeak = line.split(":", 1)[1].strip()
            continue
        if line.startswith("Ridge:"):
            ridge = line.split(":", 1)[1].strip()
            continue
        match = name_re.match(line)
        if not match:
            continue
        name, m_dim, n_dim, tail = match.groups()
        if name == "00_cpu_ref":
            continue
        metrics = metrics_re.search(tail)
        if not metrics:
            continue
        ms, gflops, gbps, ai, occ = metrics.groups()
        rows.append(
            {
                "kernel": name,
                "size": f"{m_dim}x{n_dim}",
                "ms": float(ms),
                "gflops": float(gflops),
                "gbps": float(gbps),
                "ai": float(ai),
                "occ": occ,
            }
        )

rows.sort(key=lambda r: (r["kernel"], int(r["size"].split("x")[0])))

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with OUT_FILE.open("w", encoding="utf-8") as out:
    out.write("# Detailed Performance Results\n\n")
    if device:
        out.write(f"* Device: {device}\n")
    if fpeak and bpeak and ridge:
        out.write(f"* Peaks: Fpeak {fpeak}, Bpeak {bpeak}, Ridge {ridge}\n")
    if run_started:
        out.write(f"* Run started: {run_started}\n")
    if run_finished:
        out.write(f"* Run finished: {run_finished}\n")
    out.write("\n")
    out.write("| Kernel | Size | Time (ms) | GFLOPS | GB/s | FLOPs/byte | Occupancy |\n")
    out.write("| --- | --- | --- | --- | --- | --- | --- |\n")
    for row in rows:
        out.write(
            f"| {row['kernel']} | {row['size']} | {row['ms']:.2f} | "
            f"{row['gflops']:.1f} | {row['gbps']:.1f} | {row['ai']:.2f} | {row['occ']} |\n"
        )

print(f"Wrote {len(rows)} entries to {OUT_FILE}")
