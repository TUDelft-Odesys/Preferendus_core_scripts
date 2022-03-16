import pathlib

from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="tetra_pfm",
    version="0.1",
    description="Connect to Tetra endpoint to solve preference functions",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/HaroldPy11/PFM_core_scripts/tetra_pfm",
    author="Harold van Heukelum",
    author_email="harold.van.heukelum@boskalis.com",
    license="MIT",
    classifiers=[
        "License :: MIT :: ",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=["numpy", "urllib3", "requests"],
)
