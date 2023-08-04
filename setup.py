from setuptools import setup, find_packages

setup(
    author="Samuel JS Wright",
    description="Utilities for cleaning geochemical data",
    name="clnutils",
    version="0.1.0",
    packages=find_packages(include=["clnutils", "clnutils.*"]),
    install_requires=[
        "numpy>=1.22",
        "pandas>=1.5",
        "seaborn>=0.12",
        "matplotlib>=3.7",
    ],
    python_requires=">=3.9",
)
