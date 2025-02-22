import os
import re
from setuptools import setup, find_packages

def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "flowinn", "version.py")
    with open(version_file, "r", encoding="utf-8") as f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
        if version_match:
            return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="flowinn",
    version=get_version(),
    author="Jon Errasti Odriozola",
    author_email="errasti13@gmail.com",
    description="fl0wINN: Multi-Scale Turbulent Flow Investigation using Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/errasti13/flowINN",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26.4',
        'matplotlib>=3.8.3',
        'scipy>=1.13.1',
        'tensorflow>=2.14.1',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
