"""Setup script for loan default prediction package."""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split("\n")
    requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]

setup(
    name="loan-default-prediction",
    version="1.0.0",
    description="Production-ready ML pipeline for loan default prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ML Engineer",
    author_email="ml-engineer@example.com",
    url="https://github.com/yourusername/loan-default-prediction",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        "console_scripts": [
            "loan-train=scripts.train:main",
            "loan-predict=scripts.predict:main",
            "loan-monitor=scripts.monitor:main",
        ],
    },
)

