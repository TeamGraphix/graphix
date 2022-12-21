<img src="https://github.com/TeamGraphix/graphix/raw/master/docs/logo/black_with_name.png" alt="logo" width="600">

[![Documentation Status](https://readthedocs.org/projects/graphix/badge/?version=latest)](https://graphix.readthedocs.io/en/latest/?badge=latest)
![GitHub](https://img.shields.io/github/license/TeamGraphix/graphix)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/graphix)
![PyPI](https://img.shields.io/pypi/v/graphix)

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

Read the [tutorial](https://graphix.readthedocs.io/en/latest/tutorial.html) to learn how to use `Graphix`. We also have a few demos [here](https://graphix.readthedocs.io/en/latest/algorithms.html) (more will be added).

For theoretical background, read our quick introduction into [MBQC](https://graphix.readthedocs.io/en/latest/intro.html) and [LC-MBQC](https://graphix.readthedocs.io/en/latest/lc-mbqc.html).

## Citing

A paper will be out soon, stay tuned.

## Contributing

We use [GitHub issues](https://github.com/TeamGraphix/graphix/issues) for tracking requests and bugs. 

## Core Contributors

Dr. Shinichi Sunami (University of Oxford)

Masato Fukushima (University of Tokyo, Fixstars Amplify)

## Acknowledgements

<p><a href="https://amplify.fixstars.com/en/">
<img src="https://github.com/TeamGraphix/graphix/raw/master/docs/imgs/fam_logo.png" alt="amplify" width="200"/>
</a></p>

## License

[Apache License 2.0](LICENSE)
