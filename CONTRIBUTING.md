# Contributing

thank you for your interest in `Graphix`!

## Motivation

The aim of `graphix` is to make the measurement-based quantum computing (MBQC) accessible by creating a one-stop environment to study and research MBQC. 


## Getting started working on your contribution for `graphix`

We recommend to [fork the repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo) before working on new features, whether that being an existing issue or your own idea.
Once created, you'll need to clone the repository, and you can follow below to set up the environment. You may want to set up virtual environment, such as `conda env` or `pipenv` before setting up the environment.
```
   $ git clone git@github.com:<username>/graphix.git
   $ cd graphix
   $ pip install -r requirements.txt
   $ python setup.py develop
```

You may want to install additional packages for plotting, testing and code formatting. Specifically, `matplotlib` is necessary to run codes in the `example` folder.
```
   $ pip install black==24.4.0 isort==5.13.2 pytest matplotlib
```

Before comitting the code, make sure to format with `black` and `isort`. As they sometimes have breaking changes (that will trigger error in our CI pipeline even after formatting), we fixed the versions to `black==24.4.0` and `isort==5.13.2` as above. To format a python file, just run
```
   $ black <file path>
   $ isort <file path>
```
and you are ready to commit the changes.
Note that formatter configurations are listed in pyproject.toml.

## Creating pull request

When creating a pull request, you'll see default text already filled in to the PR comment box. Please read them and fill in the missing parts, so the motivation and the changes of the PR are clear.
Please link to the issue by mentioning the issue number (#10, for example)

## Last but not least..

Please consider giving the repository a star, as well as citing in your work!
