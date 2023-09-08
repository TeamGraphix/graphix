<img src="https://github.com/TeamGraphix/graphix/raw/master/docs/logo/black_with_name.png" alt="logo" width="550">

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/graphix)
![PyPI](https://img.shields.io/pypi/v/graphix)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg)](https://unitary.fund/)
[![DOI](https://zenodo.org/badge/573466585.svg)](https://zenodo.org/badge/latestdoi/573466585)
[![Documentation Status](https://readthedocs.org/projects/graphix/badge/?version=latest)](https://graphix.readthedocs.io/en/latest/?badge=latest)
![GitHub](https://img.shields.io/github/license/TeamGraphix/graphix)
[![Downloads](https://static.pepy.tech/badge/graphix)](https://pepy.tech/project/graphix)

**Graphix** is a measurement-based quantum computing (MBQC) compiler to generate, optimize and simulate MBQC *measurement patterns*.

## Feature

- We integrate an efficient [graph state simulator](https://graphix.readthedocs.io/en/latest/lc-mbqc.html) as an optimization routine of MBQC *measurement pattern*, with which we can classically [preprocess all Pauli measurements](https://graphix.readthedocs.io/en/latest/tutorial.html#performing-pauli-measurements) (corresponding to the elimination of all Clifford gates in the gate network - c.f. [Gottesman-Knill theorem](https://en.wikipedia.org/wiki/Gottesman–Knill_theorem) and more general pattern rewriting method based on [ZX diagram rewriting](https://arxiv.org/abs/2003.01664)), reducing the required size of graph state to run the computation.
- We implement tensor-network simulation backend for MBQC with which thousands of qubits (graph nodes) can be simulated with modest computing resources (e.g. laptop), without approximation.
- We are developing density matrix simulation backend for noisy MBQC simulations with customizable noise models.

## Installation
Install `graphix` with `pip`:

```bash
$ pip install graphix
```

Install together with device interface:
```bash
$ pip install graphix[extra]
```
this will install `graphix` and [IBMQ interface](https://github.com/TeamGraphix/graphix-ibmq) to run MBQC patterns on IBM devices and Aer simulator.

We are currently adding more quantum device interfaces.
Please suggest in [issues](https://github.com/TeamGraphix/graphix/issues) if you have any particular device in mind!


## Next Steps

- We have a few [demos](https://graphix.readthedocs.io/en/latest/gallery/index.html) showing basic usages of `Graphix`.
- You can run demos on your browser:
  - Preprocessing Clifford gates: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TeamGraphix/graphix-examples/HEAD?labpath=deutsch-jozsa.ipynb)
  - Using tensor-network simulator backend: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TeamGraphix/graphix-examples/HEAD?labpath=qft_with_tn.ipynb)
  - QAOA circuit: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TeamGraphix/graphix-examples/HEAD?labpath=qaoa.ipynb)

- Read the [tutorial](https://graphix.readthedocs.io/en/latest/tutorial.html) for more comprehensive guide.

- For theoretical background, read our quick introduction into [MBQC](https://graphix.readthedocs.io/en/latest/intro.html) and [LC-MBQC](https://graphix.readthedocs.io/en/latest/lc-mbqc.html).

## Citing

> Shinichi Sunami and Masato Fukushima, Graphix. (2023) https://doi.org/10.5281/zenodo.7861382

Update on the [arXiv paper](https://arxiv.org/pdf/2212.11975.pdf): [^1]

[^1]: Following the release of this arXiv preprint, we were made aware of [Backens et al.](https://quantum-journal.org/papers/q-2021-03-25-421/) and related work, where graph-theoretic simplification (Pauli measurement elimination) of patterns were shown.
Many thanks for letting us know about this work - at the time of the writing we were not aware of these important relevant works but will certainly properly mention in the new version; we are working on significant restructuring and rewriting of the paper and hope to update the paper this autumn.

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
