import io
from setuptools import setup, find_packages

with io.open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="linear_regression_model",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Linear regression implementation from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ankan62/linear_regression_model",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "numpy>=1.20.0",
    ],
)
