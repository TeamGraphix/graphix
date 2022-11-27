![Image](docs/logo/black_with_text.png)

**Graphix** is an open-source library to generate, optimize and simulate the measurement-based quantum computing (MBQC). 

## Feature

- We integrate efficient stabilizer simulator as an optimization routine of MBQC *measurement pattern*, with which we can classically preprocess the MBQC to eliimnate all Pauli measurements (corresponding to elimination of all Clifford gates in the gate network - c.f. Gottesman-Knill theorem).
- We implement Matrix Product State (MPS) simulation of MBQC with which thousands of qubits (graph nodes) can be simulated with modest computing resource (e.g. laptop), without approximation.
- Our pattern-based construction and the optimization routines are suitable for high-level optimization to run quantum algorithms on quantum hardware with minimal resource state size requirements. We plan to add quantum hardware and hardware emulators as pattern execution backends.

## Installation
Install `graphix` with `pip`:

```bash
$ pip install graphix
```

## Contributing

We use [GitHub issues](https://github.com/Graphix/graphix/issues) for tracking requests and bugs. 

## Next Steps

See `examples` folder and read the [documentation]().

## Authors

Shinichi Sunami (University of Oxford)

Masato Fukushima (University of Tokyo)

## Citing

A paper will be out soon, stay tuned.

## License

[Apache License 2.0](LICENSE.txt)
