# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2023-04-25

### Changed

- Move import path of `generate_from_pattern` from `graphix.gflow` to `grahpix.generator` (#40)
- Rename `Pattern.get_measurement_order` to `Pattern.get_measurement_commands` (#40)
- Modify `Pattern.get_meas_plane` method to work for Clifford-decorated nodes (#40)

### Fixed

- Fix QFT circuits in examples (#38)
- Fix the stability issue of `Pattern.minimize_space` method which sometimes failed to give theoretical minimum space for patterns with flow (#40)


## [0.2.0] - 2023-03-16

### Added

- Fast circuit translation for some types gates and circuits (see PR) (#16)
- Additional required modules: `quimb` and `autoray` for more performant TN backend (#32)

### Changed

- Restructured tensor-network simulator backend for more optimized contraction (#32)
- Modify TN simulator interface to `TensorNetwork` from `MPS` (#32)

### Fixed

- Treatment of isolated node in `perform_pauli_measurements()` method (#36)


## [0.1.2] - 2022-12-21

### Added

- added QAOA demo to documentation and improved readme

### Fixed

- Fix manual input pattern (#11)


## [0.1.1] - 2022-12-19

### Fixed

- nested array error in numpy 1.24 (deprecated from 1.23.*) fixed and numpy version changed in requirements.txt (#7)
- circuit.standardize_and_transpile() error fixed (#9)


## [0.1.0] - 2022-12-15
