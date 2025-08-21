from setuptools import setup, find_packages
import re
import os

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open("dcfoil/__init__.py").read(),
)[0]

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="dcfoil",
    version=__version__,
    description="DCFoil is package for hydrofoil analysis and optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="hydrofoil",
    author="",
    author_email="",
    url="https://github.com/mdolab/DCFoil.jl",
    license="Apache License Version 2.0",
    packages=find_packages(include=["dcfoil*"]),
    install_requires=[
        "numpy>=1.16",
        "packaging",
    ],
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
