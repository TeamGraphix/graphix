"""
MQTBench simulator benchmarking
===============================

Benchmark Graphix simulation performance using quantum circuits from MQTBench.

The script:
    1. Fetches benchmark circuits from MQT Bench (ghz, graphstate, dj).
    2. Parses OpenQASM 3 into Graphix circuits via graphix-qasm-parser.
    3. Transpiles to MBQC patterns, standardizes and minimizes space.
    4. Measures execution time of :meth:`graphix.pattern.Pattern.simulate_pattern`
       with backend="statevector".
    5. Plots Time vs. Number of Qubits per algorithm and saves to mqtbench_results.png.
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt

try:
    from graphix_qasm_parser import OpenQASMParser
except ImportError:
    print("This script requires graphix-qasm-parser. Install with: pip install graphix-qasm-parser")
    sys.exit(1)

try:
    from mqt.bench import BenchmarkLevel, get_benchmark
except ImportError:
    print("This script requires mqt.bench. Install with: pip install mqt.bench")
    sys.exit(1)

try:
    import qiskit.qasm3
except ImportError:
    print("This script requires qiskit (for OpenQASM 3 export). Install with: pip install qiskit")
    sys.exit(1)

from graphix import Circuit

# Benchmark configuration
ALGORITHMS = ["ghz", "graphstate", "dj"]
QUBIT_COUNTS = range(2, 21)


def run_benchmarks():
    """Fetch MQTBench circuits, parse to Graphix, transpile, simulate and collect timings."""
    results = {algo: [] for algo in ALGORITHMS}
    qubit_counts_by_algo = {algo: [] for algo in ALGORITHMS}

    for algo in ALGORITHMS:
        for n_qubits in QUBIT_COUNTS:
            try:
                qc = get_benchmark(algo, BenchmarkLevel.ALG, n_qubits)
                qasm_string = qiskit.qasm3.dumps(qc)
            except Exception as e:
                print(f"Skipping {algo} n={n_qubits}: get_benchmark/qasm export failed: {e}")
                continue

            try:
                parser = OpenQASMParser()
                circuit = parser.parse_str(qasm_string)
            except Exception as e:
                print(f"Skipping {algo} n={n_qubits}: parse failed: {e}")
                continue

            if not isinstance(circuit, Circuit):
                print(f"Skipping {algo} n={n_qubits}: parser did not return a Graphix Circuit")
                continue

            try:
                transpile_result = circuit.transpile()
                pattern = transpile_result.pattern
                pattern.standardize()
                pattern.minimize_space()
            except Exception as e:
                print(f"Skipping {algo} n={n_qubits}: transpile failed: {e}")
                continue

            start = perf_counter()
            try:
                pattern.simulate_pattern(backend="statevector")
            except Exception as e:
                print(f"Skipping {algo} n={n_qubits}: simulate failed: {e}")
                continue
            end = perf_counter()
            elapsed = end - start

            results[algo].append(elapsed)
            qubit_counts_by_algo[algo].append(n_qubits)
            print(f"{algo} n={n_qubits}: {elapsed:.3f}s")

    return results, qubit_counts_by_algo


def plot_and_save(results, qubit_counts_by_algo):
    """Plot Time vs. Qubits per algorithm and save to benchmarks/mqtbench_results.png."""
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for algo in ALGORITHMS:
        qubits = qubit_counts_by_algo[algo]
        times = results[algo]
        if qubits and times:
            ax.scatter(qubits, times, label=algo, marker="o")

    ax.set(
        xlabel="Number of qubits",
        ylabel="time (s)",
        yscale="log",
        title="MQTBench simulation time (Graphix statevector)",
    )
    fig.legend(bbox_to_anchor=(0.85, 0.9))

    out_path = Path(__file__).resolve().parent / "mqtbench_results.png"
    fig.savefig(out_path)
    print(f"Plot saved to {out_path}")


def main():
    results, qubit_counts_by_algo = run_benchmarks()
    plot_and_save(results, qubit_counts_by_algo)


if __name__ == "__main__":
    main()
