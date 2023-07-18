import os
from glob import glob
from setuptools import setup, find_packages

exec(open("sol-kde/version.py").read())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sol-kde",
    version=__version__,
    description="A k-deep ensemble model to predict solubility",
    author="Mayk Caldas",
    author_email="andrew.white@rochester.edu",
    url="https://github.com/maykcaldas/molsol",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'tensorflow',
        'kdeepensemble',
        'sklearn',
        'rdkit',
        'selfies',
    ],
    test_suite="tests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)