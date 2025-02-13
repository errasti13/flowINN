from setuptools import setup, find_packages

setup(
    name="flowinn",
    version="1.1.0",
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.14.1',
        'numpy>=1.26.4',
        'matplotlib>=3.8.3',
        'scipy>=1.13.1',
    ],
    author="Jon Errasti Odriozola",
    author_email="errasti13@gmail.com",
    description="fl0wINN: Multi-Scale Turbulent Flow Investigation using Neural Networks",
    keywords="physics-informed neural networks, computational fluid dynamics, turbulence modeling",
    python_requires='>=3.8',
)
