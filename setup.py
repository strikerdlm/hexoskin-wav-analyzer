from setuptools import setup, find_packages

setup(
    name="hexoskin-wav-analyzer",
    version="0.0.3",
    description="Tool for analyzing Hexoskin WAV files containing physiological data",
    author="Diego Malpica MD",
    author_email="dlmalpicah@unal.edu.co",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "seaborn>=0.11.0",
        "openpyxl>=3.0.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "hexoskin-analyzer=hexoskin_analyzer.hexoskin_wav_loader:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
) 