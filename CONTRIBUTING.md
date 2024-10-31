# Contributing

thank you for your interest in `Graphix`!

## Motivation

The aim of `graphix` is to make the measurement-based quantum computing (MBQC) accessible by creating a one-stop environment to study and research MBQC.

## Getting started working on your contribution for `graphix`

We recommend to [fork the repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo) before working on new features, whether that being an existing issue or your own idea.
Once created, you'll need to clone the repository, and you can follow below to set up the environment. You may want to set up virtual environment, such as `conda env` or `pipenv` before setting up the environment.

```bash
git clone git@github.com:<username>/graphix.git
cd graphix
pip install -e .[dev]
```

You may want to install additional packages.
Specifically, `matplotlib` is necessary to run codes in the `example` folder.

```bash
pip install matplotlib
```

For other depencencies for the docs build, see `docs/requirements.txt`.

Before comitting the code, make sure to format with `ruff`.
To format a python file, just run in the top level of the repository:

```bash
# Be sure to use the latest version of ruff
pip install -U ruff
# Sort imports and format
ruff check --select I --fix .
ruff format .
```

and you are ready to commit the changes.

## Creating pull request

When creating a pull request, you'll see default text already filled in to the PR comment box. Please read them and fill in the missing parts, so the motivation and the changes of the PR are clear.
Please link to the issue by mentioning the issue number (#10, for example)

## Also..

Please consider giving the repository a star, as well as citing in your work!

<br>
<br>

# Creating maintainable codebase (information only)

[This is not related to the contribution guide, but rather a premature guideline for a better code quality that is to be updated regularly] <br>

Here are the list of thoughts we have for improving maintainability of `graphix` as of mid-2024. <br>
We'd like to hear your inputs on the blow, feel free to create issues / contact maintainers to discuss more!


### Maintenability

- Reduce public API
  - Hide implementation details
  - Prefer private module/method/attribute
  - Drop unmaintained features
  - Downsize the codebase
  - Wisely use inheritance to simplify the code
- Reduce external dependencies
- Decouple/simplify module dependencies
  - No circular import
- Eagerly use type hinting
  - Ensure type safety statically
- Use CI
  - Linter/formatter/type checker
  - Automated test
  - Coverage report

### Performance

- Drop slow API added just for convenience
  - Make users choose appropriate backend
- Avoid common pitfalls
  - ex. `list` abuse
- Use appropriate internal data structure
  - "how it looks" and "how it is implemented" can differ
  - Rely on `numpy`/`numba`

### Usability

- Do not expose internal data representation
- Use coherent naming
  - Use overload
- Prefer pythonic design
  - Define dunders
  - Prefer property over getter/setter
  - Stick to PEP8
- Type hint as documentation
- Safe API
  - Fail-fast
  - Copy whenever necessary


### To-do list:
To achieve the above and make this a well-organized package, we are working on the topics below, which will be reflected in issues and PRs.

 - [x] Start enforcing linting/formatting
 - [x]  - `ruff`/`mypy`/`pyright`/`isort
 - [x] Various CI integration
 - [x] Start adding type annotations
 - [x] Refactor upstream modules
   - e.g. `graphix.linalg_validations`
 - [ ] Refactor core modules
   - e.g. `graphix.pauli`
 - [ ] Improve test coverage
 - [ ] Hide implementation details
   - e.g. `graphix._db`
 - [ ] Optimize performance
 - [ ] Enhance documentation
 - [ ] Refactor
   - Remove duplication
   - Resolve circular import
   - Safe API
 - [ ] Make linter more strict
 - [ ] General refactoring
   - Especially graph-related modules
 - [ ] Locate bottlenecks and optimize
   - `cProfile` + `snakeviz`
 - [ ] Improve usability
   - e.g. Add pattern syntax checker
 - [ ] Improve generality
   - e.g. Allow anything hashable as graph node
 - [ ] Make `graphix` perfectly-typed
 - [ ] Refactor high-level modules
   - ex. `graphix.pattern`
 - [ ] Refactor/optimize simulator
   - May need throrough reimplementation
 - [ ] Update related packages
   - `graphix-ibmq`/`graphix-perceval`