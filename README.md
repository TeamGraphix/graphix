<p align="center">
  <picture>
    <source
      media="(prefers-color-scheme: dark)"
      srcset="assets/white_with_name.png"
    >
    <source
      media="(prefers-color-scheme: light)"
      srcset="assets/black_with_name.png"
    >
    <img
      src="assets/black_with_name.png"
      alt="Graphix logo"
      width="550"
    >
  </picture>
</p>


![PyPI](https://img.shields.io/pypi/v/graphix)
![License](https://img.shields.io/github/license/TeamGraphix/graphix)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/graphix)
[![Downloads](https://static.pepy.tech/badge/graphix)](https://pepy.tech/project/graphix)
[![Unitary Foundation](https://img.shields.io/badge/Supported%20By-Unitary%20Foundation-brightgreen.svg)](https://unitary.foundation/)
[![DOI](https://zenodo.org/badge/573466585.svg)](https://zenodo.org/badge/latestdoi/573466585)
[![CI](https://github.com/TeamGraphix/graphix/actions/workflows/ci.yml/badge.svg)](https://github.com/TeamGraphix/graphix/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/TeamGraphix/graphix/graph/badge.svg?token=E41MLUTYXU)](https://codecov.io/gh/TeamGraphix/graphix)
[![Documentation Status](https://readthedocs.org/projects/graphix/badge/?version=latest)](https://graphix.readthedocs.io/en/latest/?badge=latest)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Graphix** is an open-source Python library for building, optimizing, and simulating measurement-based quantum computations (MBQC). It features:

- The measurement calculus framework with integrated graphical rewrite rules for pre-processing Pauli measurements.
- Circuit-to-pattern transpiler, automatic pattern generation from graph states and manual pattern generation.
- State-of-the-art causal flow, gflow and Pauli flow finding algorithms and visualization tools.
- Statevector and density matrix simulation backends.
- Circuit extraction.
- Parametric MBQC computations.
- Export to OpenQASM for external simulation and QPU integration.
- An extensible [plugin ecosystem](#graphix-plugins).

For usage examples drawn from recent research on MBQC and further details on the architecture, design and algorithm implementations, check our [latest preprint](https://arxiv.org/abs/2608.24781)!

Full documentation, including detailed Graphix tutorials and an introduction to measurement-based quantum computation (MBQC), is available at [graphix.readthedocs.io](https://graphix.readthedocs.io/en/latest/).



## Installation

Install `graphix` with `pip`:

```bash
pip install graphix
```

### Editable installation using pip and uv

As well as support for editable installation using pip, Graphix also supports using `uv`:

```bash
git clone https://github.com/TeamGraphix/graphix.git
cd graphix
uv sync --extra dev
```

This creates a virtual environment and installs all development dependencies from the `pyproject.toml` and `uv` lockfile.

## Using Graphix

### Generating a pattern from a circuit

```python
from graphix import Circuit

circuit = Circuit(4)
...
pattern = circuit.transpile().pattern
pattern.standardize()
pattern.shift_signals()

pattern.draw_graph(flow_from_pattern=False)
```

<img src="assets/qaoa_pattern.png" alt="graph_flow" width="900">

<small>See [our example code](examples/qaoa.py) to generate this pattern. Arrows indicate the [_causal flow_](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.74.052310) of MBQC and dashed lines are the other edges of the graph. The vertical dashed partitions and the labels below indicate the execution _layers_ based on the partial order associated with the (maximally-delayed) flow of the graph. Measurements happen from left to right, and nodes in the same layer can be measured simultaneously. </small>

### Pre-processing Pauli measurements (Clifford gates)

```python
pattern.infer_pauli_measurements()
pattern.remove_pauli_measurements()

pattern.draw(flow_from_pattern=False)
```
<img src="assets/qaoa_pauli_removed.png" alt="graph_gflow" width="340">

<small>(here, the visualization is based on [_generalized flow_](https://iopscience.iop.org/article/10.1088/1367-2630/9/8/250)).</small>

### Simulating the pattern

```python
state_out = pattern.simulate()
```

### And more...

Check [graphix.readthedocs.io](https://graphix.readthedocs.io/en/latest/) !

## Graphix plugins

- [`graphix-ibmq`](https://github.com/TeamGraphix/graphix-ibmq): pattern transpiler for IBMQ / `qiskit`
- [`graphix-perceval`](https://github.com/TeamGraphix/graphix-perceval): pattern transpiler for Quandela's `perceval` simulator and QPU
- [`graphix-pyzx`](https://github.com/TeamGraphix/graphix-pyzx): conversion between open graphs and PyZX diagrams
- [`graphix-qasm-parser`](https://github.com/TeamGraphix/graphix-qasm-parser): a plugin for parsing OpenQASM circuit.
- [`graphix-stim-backend`](https://github.com/TeamGraphix/graphix-stim-backend): `stim` backend for efficient Clifford pattern simulation
- [`graphix-stim-compiler`](https://github.com/TeamGraphix/graphix-stim-compiler): `stim` backend for efficient compilation of Clifford maps.
- [`graphix-symbolic`](https://github.com/TeamGraphix/graphix-symbolic): parameterized patterns with symbolic simulation
- [`graphix-mqtbench`](https://github.com/TeamGraphix/graphix-mqtbench): an interface between Graphix and the [MQT Bench](https://mqt.readthedocs.io/projects/bench/en/latest/index.html) suite for benchmarking purposes.

## Projects using Graphix

- [veriphix](https://github.com/qat-inria/veriphix): verified blind quantum computation and benchmarking.
- [optyx](https://github.com/quantinuum-dev/optyx): ZX-based software for networked quantum computing.
- [QPatLib](https://github.com/vws100/QPatLib): a community repository for MBQC patterns and supporting tooling.

## Citing

If you use Graphix in your research, please consider citing the [associated papers](https://arxiv.org/abs/2608.24781) and the accompanying [software release](https://zenodo.org/records/21997813).

### Papers
```
@article{uldemolins2026_graphix,
  title={Graphix: A software framework for Measurement-Based Quantum Computation},
  author={Uldemolins, Mateo and Nair, Pranav and Graham, Emlyn and Sunami, Shinichi and Martinez, Thierry and Garnier, Maxime},
  journal={arXiv preprint arXiv:2608.24781},
  year={2026}
}
```
```
@article{sunami2022_graphix,
  title={Graphix: Optimizing and simulating measurement-based quantum computation on local-Clifford decorated graph},
  author={Sunami, Shinichi and Fukushima, Masato},
  journal={arXiv preprint arXiv:2212.11975},
  year={2022}
}
```

### Software
```
@software{graphix_v04,
  author    = {Uldemolins, Mateo and Fukushima, Masato and
              Graham, Emlyn and Nair, Pranav and Sasaki, Daichi and
              Shiratani, Sora and Watanabe, Yuki and
              Martinez, Thierry and Garnier, Maxime and Sunami, Shinichi},
  title     = {Graphix},
  version   = {0.4},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21997813},
}
```

## Contributing

We use [GitHub issues](https://github.com/TeamGraphix/graphix/issues) for tracking feature requests and bug reports.

## Discussion channels

- Our Slack channel, for regular discussions and questions: https://graphix-org.slack.com

- Please visit [Unitary Foundation's Discord server](https://discord.com/servers/unitary-foundation-764231928676089909), where you can find a channel for `graphix`.

## Current maintainers (alphabetical order)

- [Maxime Garnier](https://github.com/mgarnier59) (Inria Paris)
- [Thierry Martinez](https://github.com/thierry-martinez) (Inria Paris)
- [Pranav Nair](https://github.com/pranav97nair) (Inria Paris)
- [Mateo Uldemolins](https://github.com/matulni) (Inria Paris)

## Acknowledgements

Graphix was started in 2022 by [Shinichi Sunami](https://github.com/shinich1) (University of Oxford) and [Masato Fukushima](https://github.com/masa10-fk) (University of Tokyo, Fixstars Amplify). [Daichi Sasaki](https://github.com/d1ssk), Yuki Watanabe, and [Sora Shiratani](https://github.com/EarlMilktea) (University of Tokyo, Fixstars Amplify) later joined the team.

In 2023, the [QAT team](https://qat.inria.fr/presentation/) joined the effort and now maintains the library. We thank Emlyn Graham (Inria Paris) for his contributions to Graphix from 2025 to 2026.

The project has been supported by [Fixstars Amplify](https://amplify.fixstars.com/en/) and the [Unitary Foundation](https://unitary.foundation/), and currently participates in the [French Hybrid Quantum Initiative](https://www.hqi.fr/).



<p align="center">
<a href="https://unitary.foundation/grants/" style="margin-right: 30px;">
<img src="https://unitary.foundation/images/UFoundation.png" alt="Unitary Foundation logo" width="180"/>
</a>
<a href="https://amplify.fixstars.com/en/" style="margin-right: 30px;">
<img src="https://github.com/user-attachments/assets/ffbf7ff6-14b8-448c-86a1-39583f30a0f4" alt="Fixstars Amplify logo" width="230"/>
</a>
</p>


<p align="center">
<a href="https://www.inria.fr/" style="margin-right: 30px;">
  <img src="https://www.inria.fr/sites/default/files/2025-04/RF-Inria_Logo_RVB.jpg" alt="Inria logo" width="150"/>
</a>
&nbsp;
<a href="https://www.ens.psl.eu/" style="margin-right: 30px;">
  <img src="https://www.ens.psl.eu/sites/default/files/logo_ens_psl_en_png.png" alt="ENS PSL logo" width="150"/>
</a>
&nbsp;
<a href="https://qat.inria.fr/presentation/" style="margin-right: 30px;">
  <img src="https://qat.inria.fr/assets/icons/icon-512x512.png" alt="QAT logo" width="90"/>
</a>
</p>

<p align="center">
<a href="https://www.hqi.fr">
  <img src="https://www.hqi.fr/wp-content/uploads/2022/06/logo_HQI_RVB.jpg" alt="HQI logo" width="150"/>
</a>
</p>


## License

[Apache License 2.0](LICENSE)
