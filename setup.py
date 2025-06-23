from setuptools import setup, find_packages

setup(
    name="ef_tss",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.0.1",
        "scipy>=1.13.1",
    ],
    python_requires='>=3.9',
)