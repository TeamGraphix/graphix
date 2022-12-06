![Image](docs/logo/black_with_text.png)

[![Documentation Status](https://readthedocs.org/projects/graphix/badge/?version=latest)](https://graphix.readthedocs.io/en/latest/?badge=latest)
![GitHub](https://img.shields.io/github/license/TeamGraphix/graphix)

**Graphix** is an open-source library to optimize and simulate measurement-based quantum computing (MBQC). 

## Feature

- We integrate an [efficient stabilizer simulator](graphix/graphsim) as an optimization routine of MBQC *measurement pattern*, [with which we can classically preprocess all Pauli measurements]() (corresponding to the elimination of all Clifford gates in the gate network - c.f. [Gottesman-Knill theorem](https://en.wikipedia.org/wiki/Gottesman–Knill_theorem)), significantly reducing the required size of graph state to run the computation (typically by a factor of 3 or more).
- We implement Matrix Product State (MPS) simulation of MBQC with which thousands of qubits (graph nodes) can be simulated with modest computing resources (e.g. laptop), without approximation.
- Our pattern-based construction and optimization routines are suitable for high-level optimization to run quantum algorithms on MBQC quantum hardware with minimal resource state size requirements. We plan to add quantum hardware emulators (and quantum hardware) as pattern execution backends.

## Installation
<!-- Install `graphix` with `pip`:

```bash
$ pip install graphix
``` -->

`clone` the repository and 

```bash
$ python setup.py install
```

first version will be available on pypi soon, for `pip install`.

## Contributing

We use [GitHub issues](https://github.com/TeamGraphix/graphix/issues) for tracking requests and bugs. 

## Next Steps

Read the [tutorial](https://graphix.readthedocs.io/en/latest/tutorial.html).

## Core Contributors

Shinichi Sunami (University of Oxford)

Masato Fukushima (University of Tokyo, [Fixstars Amplify](https://amplify.fixstars.com/en/))

## Citing

A paper will be out soon, stay tuned.

## License

[Apache License 2.0](LICENSE)
