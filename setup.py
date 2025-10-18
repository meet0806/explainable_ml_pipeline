"""
Setup script for Explainable ML Pipeline
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="explainable-ml-pipeline",
    version="1.0.0",
    author="ML Systems Team",
    author_email="team@example.com",
    description="Explainable ML Pipelines with Agentic AI for Healthcare and Finance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/explainable-ml-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ml-pipeline=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config.yaml"],
    },
)

