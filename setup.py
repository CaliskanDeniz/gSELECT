from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gSELECT",
    version="0.1.0",
    author="Deniz Caliskan",
    author_email="caliskandeniz@outlook.de",
    description="A package for single-cell gene selection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/caliskandeniz/gSELECT",
    packages=find_packages(),
    install_requires=[
        "scanpy",
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "polars",
        "scipy",
        "principal_feature_analysis"
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ],
)
