"""Setup script for the Supernote Math project."""

from setuptools import setup, find_packages

setup(
    name="supernote-math",
    version="0.1.0",
    description="Math recognition and solving for Supernote Nomad",
    author="Julian",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.10.0",
        "numpy>=1.22.0",
        "scipy>=1.8.0",
        "opencv-python>=4.6.0",
        "matplotlib>=3.5.0",
        "sympy>=1.11.0",
        "latex2sympy2>=0.3.6",
        "pillow>=9.0.0",
        "tqdm>=4.64.0",
        "pandas>=1.5.0",
        "typer>=0.7.0",
        "pyyaml>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "jupyter>=1.0.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
        ],
    },
)