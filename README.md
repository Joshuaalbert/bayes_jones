[![Python](https://img.shields.io/pypi/pyversions/bayes_jones.svg)](https://badge.fury.io/py/bayes_jones)
[![PyPI](https://badge.fury.io/py/bayes_jones.svg)](https://badge.fury.io/py/bayes_jones)
[![Documentation Status](https://readthedocs.org/projects/bayes_jones/badge/?version=latest)](https://bayes_jones.readthedocs.io/en/latest/?badge=latest)

Main
Status: ![Workflow name](https://github.com/JoshuaAlbert/bayes_jones/actions/workflows/unittests.yml/badge.svg?branch=main)

Develop
Status: ![Workflow name](https://github.com/JoshuaAlbert/bayes_jones/actions/workflows/unittests.yml/badge.svg?branch=develop)

## Mission: _To make ionospheric calibration **faster, easier, and more powerful**_

# What is it?

Bayes is:

1) a set of tools for Bayesian inference of Jones matrices using JAXNS as the engine;
2) coded in JAX in a manner that allows lowering the entire inference algorithm to XLA primitives, which are
   JIT-compiled for high performance

# Documentation

You can read the documentation [here](https://bayes_jones.readthedocs.io/en/latest/#).

# Install

**Notes:**

1. BayesJones requires >= Python 3.8.
2. It is always highly recommended to use a unique virtual environment for each project.
   To use `miniconda`, have it installed, and run

```bash
# To create a new env, if necessary
conda create -n bayes_jones_py python=3.11
conda activate bayes_jones_py
```

## For end users

Install directly from PyPi,

```bash
pip install bayes_jones
```

## For development

Clone repo `git clone https://www.github.com/JoshuaAlbert/bayes_jones.git`, and install:

```bash
cd bayes_jones
pip install -r requirements.txt
pip install -r requirements-tests.txt
pip install -r requirements-examples.txt
pip install .
```


# Quick start

Checkout the examples [here](https://bayes_jones.readthedocs.io/en/latest/#).

# Change Log

14 Dec, 2023 -- BayesJones 0.0.1 released for SaLF 9 conference.

## Star History

<a href="https://star-history.com/#joshuaalbert/bayes_jones&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=joshuaalbert/bayes_jones&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=joshuaalbert/bayes_jones&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=joshuaalbert/bayes_jones&type=Date" />
  </picture>
</a>
