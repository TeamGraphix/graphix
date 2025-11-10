<img src="https://github.com/TeamGraphix/graphix/raw/master/docs/logo/black_with_name.png" alt="logo" width="550">

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

**Graphix** is a measurement-based quantum computing (MBQC) software package, featuring

- the measurement calculus framework with integrated graphical rewrite rules for Pauli measurement preprocessing
- circuit-to-pattern transpiler, graph-based deterministic pattern generator and manual pattern generation
- flow, gflow and pauliflow finding tools and graph visualization based on flows (see below)
- statevector, density matrix and tensornetwork pattern simulation backends
- QPU interface and fusion network extraction tool
- _new_: [efficient implementation of fast O(N^3) pauli-flow finding algorithm](https://github.com/TeamGraphix/graphix/pull/337)

## Installation

Install `graphix` with `pip`:

```bash
pip install graphix
```

Install together with [extra packages](https://github.com/TeamGraphix/graphix/blob/master/requirements-extra.txt):

```bash
pip install graphix[extra]
```


## Using graphix

### generating pattern from a circuit

```python
from graphix import Circuit

circuit = Circuit(4)
circuit.h(0)
...
pattern = circuit.transpile().pattern
pattern.standardize()
pattern.shift_signals()
pattern.draw_graph(flow_from_pattern=False)
```

<img src="https://github.com/TeamGraphix/graphix/assets/33350509/de17c663-f607-44e2-945b-835f4082a940" alt="graph_flow" width="750">

<small>See [our example code](examples/qaoa.py) to generate this pattern. Arrows indicate the [_causal flow_](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.74.052310) of MBQC and dashed lines are the other edges of the graph. the vertical dashed partitions and the labels 'l:n' below indicate the execution _layers_ or the order in the graph (measurements should happen from left to right, and nodes in the same layer can be measured simultaneously), based on the partial order associated with the (maximally-delayed) flow. </small>

### preprocessing Pauli measurements (Clifford gates)

```python
pattern.perform_pauli_measurements()
pattern.draw_graph()
```

<img src="https://github.com/TeamGraphix/graphix/assets/33350509/3c30a4c9-f912-4a36-925f-2ff446a07c68" alt="graph_gflow" width="140">

<small>(here, the visualization is based on [_generalized flow_](https://iopscience.iop.org/article/10.1088/1367-2630/9/8/250)).</small>

### simulating the pattern

```python
state_out = pattern.simulate_pattern(backend="statevector")
```

### and more..

- See [demos](https://graphix.readthedocs.io/en/latest/gallery/index.html) showing other features of `graphix`.
- Read the [tutorial](https://graphix.readthedocs.io/en/latest/tutorial.html) for more usage guides.
- For theoretical background, read our quick introduction into [MBQC](https://graphix.readthedocs.io/en/latest/intro.html) and [LC-MBQC](https://graphix.readthedocs.io/en/latest/lc-mbqc.html).
- Full API docs is [here](https://graphix.readthedocs.io/en/latest/references.html).

## Related packages

- [graphix-stim-backend](https://github.com/thierry-martinez/graphix-stim-backend): `stim` backend for efficient Clifford pattern simulation
- [graphix-symbolic](https://github.com/TeamGraphix/graphix-symbolic): parameterized patterns with symbolic simulation
- [graphix-ibmq](https://github.com/TeamGraphix/graphix-ibmq): pattern transpiler for IBMQ / `qiskit`
- [graphix-perceval](https://github.com/TeamGraphix/graphix-perceval): pattern transpiler for Quandela's `perceval` simulator and QPU
- [graphix-qasm-parser](https://github.com/TeamGraphix/graphix-qasm-parser): a plugin for parsing OpenQASM circuit.

### Projects using `graphix`

- [Veriphix](https://github.com/qat-inria/veriphix): verified blind quantum computation and benchmarking.
- [optyx](https://github.com/quantinuum-dev/optyx): ZX-based software for networked quantum computing

## Citing

> Zenodo: https://doi.org/10.5281/zenodo.7861382
>
> arXiv: https://doi.org/10.48550/arXiv.2212.11975

## Contributing

We use [GitHub issues](https://github.com/TeamGraphix/graphix/issues) for tracking feature requests and bug reports.

## Discussion channels

- Our Slack channel, for regular discussions and questions: https://graphix-org.slack.com

- Please visit [Unitary Foundation's Discord server](https://discord.com/servers/unitary-foundation-764231928676089909), where you can find a channel for `graphix`.

## Maintainers (alphabetical order)

- Masato Fukushima (University of Tokyo, Fixstars Amplify)
- Maxime Garnier (Inria Paris)
- Emlyn Graham (Inria Paris)
- Thierry Martinez (Inria Paris)
- Pranav Nair (Inria Paris)
- Sora Shiratani (University of Tokyo, Fixstars Amplify)
- Shinichi Sunami (University of Oxford)
- Mateo Uldemolins (Inria Paris)

## Acknowledgements

Graphix is developed partly by the Qode group of the [QAT](https://qat.inria.fr/presentation/) team, co-hosted by [Inria](https://www.inria.fr/) and [ENS](https://www.ens.psl.eu/).

Special thanks to [Fixstars Amplify](https://amplify.fixstars.com/en/), [HQI](https://www.hqi.fr) and [Unitary Foundation](https://unitary.foundation/grants.html).

<p><a href="https://amplify.fixstars.com/en/">
<img src="https://github.com/user-attachments/assets/ffbf7ff6-14b8-448c-86a1-39583f30a0f4" alt="Fixstars Amplify logo" width="230"/>
</a></p>

<p>
<a href="https://www.inria.fr/">
<img src="https://www.inria.fr/sites/default/files/2025-04/RF-Inria_Logo_RVB.jpg" alt="Inria logo" width="150"/>
</a>
<a href="https://www.ens.psl.eu/">
<img src="https://www.ens.psl.eu/sites/default/files/logo_ens_psl_en_png.png" alt="ENS PSL logo" width="150"/>
</a>
<a href="https://qat.inria.fr/presentation/">
<img src="https://qat.inria.fr/assets/icons/icon-512x512.png" alt="QAT logo" width="90"/>
</a>
<a href="https://www.hqi.fr">
<img src="https://www.hqi.fr/wp-content/uploads/2022/06/logo_HQI_RVB.jpg" alt="HQI logo" width=150"/>
</a>
</p>

<p><a href="https://unitary.foundation/grants/">
<img src="https://unitary.foundation/images/UFoundation.png" alt="Unitary Foundation logo" width="150"/>
</a></p>

## License

[Apache License 2.0](LICENSE)
