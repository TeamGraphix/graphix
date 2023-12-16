<img src="https://github.com/TeamGraphix/graphix/raw/master/docs/logo/black_with_name.png" alt="logo" width="550">

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/graphix)
![PyPI](https://img.shields.io/pypi/v/graphix)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg)](https://unitary.fund/)
[![DOI](https://zenodo.org/badge/573466585.svg)](https://zenodo.org/badge/latestdoi/573466585)
[![Documentation Status](https://readthedocs.org/projects/graphix/badge/?version=latest)](https://graphix.readthedocs.io/en/latest/?badge=latest)
![GitHub](https://img.shields.io/github/license/TeamGraphix/graphix)
[![Downloads](https://static.pepy.tech/badge/graphix)](https://pepy.tech/project/graphix)

**Graphix** is a measurement-based quantum computing (MBQC) software package, featuring
- the measurement calculus framework with integrated graphical rewrite rules for Pauli measurement preprocessing
- circuit-to-pattern transpiler, graph-based deterministic pattern generator and manual pattern generation
- flow- and gflow-based graph visualization tools
- statevector and tensornetwork pattern simulation backends
- QPU interface and fusion network extraction tool

## Installation
Install `graphix` with `pip`:

```bash
$ pip install graphix
```

Install together with device interface:
```bash
$ pip install graphix[extra]
```
this will install `graphix` and inteface for [IBMQ](https://github.com/TeamGraphix/graphix-ibmq) and [Perceval](https://github.com/TeamGraphix/graphix-perceval) to run MBQC patterns on superconducting and optical QPUs and their simulators.


## Using graphix

### Generating pattern from a circuit and inspecting the resource graph with information flow (arrows)
```python
from graphix import Circuit
circuit = Circuit(4)
circuit.h(0)
...
pattern = circuit.transpile()
pattern.draw_graph()
```
<img src="https://github.com/TeamGraphix/graphix/assets/33350509/de17c663-f607-44e2-945b-835f4082a940" alt="logo" width="750">

(QAOA circuit, see [this example](examples/qaoa.py))

### Preprocessing Pauli measurements and visualizing the resulting graph with generalized flow
```python
pattern.perform_pauli_measurements(leave_input=False)
pattern.draw_graph(show_loop=False)
```
<img src="https://github.com/TeamGraphix/graphix/assets/33350509/0153b704-7697-4c50-96d1-31fb82bf9b4d" alt="logo" width="140">

### Simulating the pattern
```python
state_out = pattern.simulate_pattern(backend='statevec')
```


### and more.. 
- See [demos](https://graphix.readthedocs.io/en/latest/gallery/index.html) showing other features of `Graphix`.
- Try demos on browser with mybinder.org: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TeamGraphix/graphix-examples/HEAD)

- Read the [tutorial](https://graphix.readthedocs.io/en/latest/tutorial.html) for more usage guides.

- For theoretical background, read our quick introduction into [MBQC](https://graphix.readthedocs.io/en/latest/intro.html) and [LC-MBQC](https://graphix.readthedocs.io/en/latest/lc-mbqc.html).

## Citing

> Shinichi Sunami and Masato Fukushima, Graphix. (2023) https://doi.org/10.5281/zenodo.7861382

Update on the [arXiv paper](https://arxiv.org/pdf/2212.11975.pdf): [^1]

[^1]: Following the release of this arXiv preprint, we were made aware of [Backens et al.](https://quantum-journal.org/papers/q-2021-03-25-421/) and related work, where graph-theoretic simplification (Pauli measurement elimination) of patterns were shown.
Many thanks for letting us know about this work - at the time of the writing we were not aware of these important relevant works but will certainly properly mention in the new version; we are working on significant restructuring and rewriting of the paper and hope to update the paper soon.

## Contributing

We use [GitHub issues](https://github.com/TeamGraphix/graphix/issues) for tracking feature requests and bugs reports. 

## Discord Server

Please visit [Unitary Fund's Discord server](https://discord.com/servers/unitary-fund-764231928676089909), where you can find a channel for `graphix` to ask questions.

## Core Contributors

Dr. Shinichi Sunami (University of Oxford)

Masato Fukushima (University of Tokyo, Fixstars Amplify)

## Acknowledgements

We are proud to be supported by [unitary fund microgrant program](https://unitary.fund/grants.html). 

<p><a href="https://unitary.fund/grants.html">
<img src="https://user-images.githubusercontent.com/33350509/233384863-654485cf-b7d0-449e-8868-265c6fea2ced.png" alt="unitary-fund" width="150"/>
</a></p>

Special thanks to Fixstars Amplify:

<p><a href="https://amplify.fixstars.com/en/">
<img src="https://github.com/TeamGraphix/graphix/raw/master/docs/imgs/fam_logo.png" alt="amplify" width="200"/>
</a></p>


## License

[Apache License 2.0](LICENSE)
