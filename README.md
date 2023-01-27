<img src="https://github.com/TeamGraphix/graphix/raw/master/docs/logo/black_with_name.png" alt="logo" width="600">

[![Documentation Status](https://readthedocs.org/projects/graphix/badge/?version=latest)](https://graphix.readthedocs.io/en/latest/?badge=latest)
![GitHub](https://img.shields.io/github/license/TeamGraphix/graphix)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/graphix)
![PyPI](https://img.shields.io/pypi/v/graphix)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg)](https://unitary.fund/)

**Graphix** is an open-source library to optimize and simulate measurement-based quantum computing (MBQC). 

## Feature

- We integrate an efficient [graph state simulator](https://graphix.readthedocs.io/en/latest/lc-mbqc.html) as an optimization routine of MBQC *measurement pattern*, with which we can classically [preprocess all Pauli measurements](https://graphix.readthedocs.io/en/latest/tutorial.html#performing-pauli-measurements) (corresponding to the elimination of all Clifford gates in the gate network - c.f. [Gottesman-Knill theorem](https://en.wikipedia.org/wiki/Gottesman–Knill_theorem)), significantly reducing the required size of graph state to run the computation.
- We implement Matrix Product State (MPS) simulation of MBQC with which thousands of qubits (graph nodes) can be simulated with modest computing resources (e.g. laptop), without approximation.
- Our pattern-based construction and optimization routines are suitable for high-level optimization to run quantum algorithms on MBQC quantum hardware with minimal resource state size requirements. We plan to add quantum hardware emulators (and quantum hardware) as pattern execution backends.

## Installation
Install `graphix` with `pip`:

```bash
$ pip install graphix
```

## Next Steps

- We have a few [demos](https://graphix.readthedocs.io/en/latest/gallery/index.html) showing basic usages of `Graphix`.
- You can run demos on your browser:
  - Preprocessing Clifford gates: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TeamGraphix/graphix-examples/HEAD?labpath=deutsch-jozsa.ipynb)
  - Using MPS simulator: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TeamGraphix/graphix-examples/HEAD?labpath=qft_with_mps.ipynb)
  - QAOA circuit: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TeamGraphix/graphix-examples/HEAD?labpath=qaoa.ipynb)

- Read the [tutorial](https://graphix.readthedocs.io/en/latest/tutorial.html) for more comprehensive guide.

- For theoretical background, read our quick introduction into [MBQC](https://graphix.readthedocs.io/en/latest/intro.html) and [LC-MBQC](https://graphix.readthedocs.io/en/latest/lc-mbqc.html).

## Citing

> S. Sunami and M. Fukushima. "Graphix: optimizing and simulating measurement-based quantum computation on local-Clifford decorated graph", 
> [arXiv:2212.11975](https://arxiv.org/abs/2212.11975) (2022).

Update on the paper: [^1]

[^1]: Following the release of this arXiv preprint, we were made aware of a previous work by [Backens et al.](https://quantum-journal.org/papers/q-2021-03-25-421/) where Pauli measurement elimination method for MBQC was developed in the context of circuit optimization. 
Many thanks for letting us know about this work, we will properly mention this work in the next version of our paper.

## Contributing

We use [GitHub issues](https://github.com/TeamGraphix/graphix/issues) for tracking requests and bugs. 

## Core Contributors

Dr. Shinichi Sunami (University of Oxford)

Masato Fukushima (University of Tokyo, Fixstars Amplify)

## Acknowledgements

We are proud to be supported by [unitary fund microgrant program](https://unitary.fund/grants.html). 

Special thanks to Fixstars Amplify:

<p><a href="https://amplify.fixstars.com/en/">
<img src="https://github.com/TeamGraphix/graphix/raw/master/docs/imgs/fam_logo.png" alt="amplify" width="200"/>
</a></p>


## License

[Apache License 2.0](LICENSE)
