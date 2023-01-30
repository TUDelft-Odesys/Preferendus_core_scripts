"""
(c) Harold van Heukelum, 2022
"""
import pathlib

from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="genetic_algorithm_pfm",
    version="1.0.0",
    description="Calculate optimal preference score of a multi-objective problem",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Graduation-project-Boskalis/Tetra_integration",
    author="Harold van Heukelum",
    author_email="harold.van.heukelum@boskalis.com",
    license="",
    classifiers=[
        "License :: :: ",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=["numpy", "matplotlib", "scipy", "requests", "pandas"],
)
