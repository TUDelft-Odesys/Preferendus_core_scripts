import pathlib

from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="PFM_core_scripts",
    version="0.1",
    description="Calculate optimal preference score of a multi-objective problem",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/HaroldPy11/PFM_core_scripts",
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
    install_requires=["numpy", "matplotlib", "scipy", "urllib3", "requests"],
)
