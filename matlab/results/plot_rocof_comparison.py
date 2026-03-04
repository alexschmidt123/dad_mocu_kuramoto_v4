#!/usr/bin/env python3
"""
Generate ROCOF comparison figure for email/report.
Run from repo root: python matlab/results/plot_rocof_comparison.py
Saves: matlab/results/rocof_comparison.png
"""
import csv
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("matplotlib not found; install with: pip install matplotlib")
    raise

REPO = Path(__file__).resolve().parent.parent.parent
MATLAB_PROBE = REPO / "matlab" / "results" / "fourteen_bus_dynamic_probe" / "observation_from_voltage.csv"
COMP_CSV = REPO / "matlab" / "results" / "comparison_table.csv"
OUT_PNG = REPO / "matlab" / "results" / "rocof_comparison.png"


def load_matlab_probe():
    out = {}
    if not MATLAB_PROBE.exists():
        return out
    with open(MATLAB_PROBE) as f:
        for r in csv.DictReader(f):
            out[int(r["bus"])] = float(r["ROCOF_max"])
    return out


def load_python_probe_bus1():
    out = {}
    if not COMP_CSV.exists():
        return out
    with open(COMP_CSV) as f:
        for r in csv.DictReader(f):
            if r["source"] == "python_probe_bus1":
                out[int(r["bus"])] = float(r["ROCOF_max"])
    return out


def main():
    matlab = load_matlab_probe()
    python = load_python_probe_bus1()
    buses = sorted(set(matlab) | set(python))
    if not buses:
        print("No data; run MATLAB and comparison script first.")
        return
    x = np.arange(len(buses))
    width = 0.35
    matlab_vals = [matlab.get(b, 0) for b in buses]
    python_vals = [python.get(b, 0) for b in buses]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, matlab_vals, width, label="MATLAB (Simulink)", color="C0")
    ax.bar(x + width / 2, python_vals, width, label="Python (swing ODE)", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels(buses)
    ax.set_xlabel("Bus")
    ax.set_ylabel("ROCOF_max (Hz/s)")
    ax.set_title("Probe at bus 1: ROCOF_max per bus — MATLAB vs Python")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
