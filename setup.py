#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup

install_requires = [
    'jax',
    'jaxlib',
    'chex',
    'matplotlib',
    'numpy',
    'tensorflow_probability',
    'jaxns>=2.3.2',
    'tomographic_kernel'
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='bayes_jones',
      version='0.0.2',
      description='Bayesian inference of Jones matrices.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/joshuaalbert/bayes_jones",
      author='Joshua G. Albert',
      author_email='albert@strw.leidenuniv.nl',
      install_requires=install_requires,
      tests_require=[
          'pytest>=2.8',
      ],
      package_dir={'': './'},
      packages=find_packages('./'),
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: OS Independent",
      ],
      python_requires='>=3.8',
      )
