from setuptools import setup, find_packages

setup(
    author="Samuel JS Wright",
    description="Utilities for cleaning geochemical data",
    name="clnutils",
    version="0.1.0",
    packages=find_packages(include=["clnutils", "clnutils.*"]),
    install_requires=[
        "pandas >= 1.5.2",
        "numpy >= 1.23.5",
        "matplotlib >= 3.7.1",
        "seaborn >= 0.12.2",
    ],
    python_requires=">=3.9",
)
