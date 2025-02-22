# fl0wINN

fl0wINN: Multi-Scale Turbulent Flow Investigation using Neural Networks

## Description

fl0wINN is a Python package for investigating multi-scale turbulent flow using neural networks. It leverages Physics-Informed Neural Networks (PINNs) to solve complex fluid dynamics problems.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Installation

You can install fl0wINN using pip:

```bash
pip install flowinn
```

## Usage

```python
from flowinn import PINN

# Example usage
pinn = PINN()
# ... your code ...
```

### Running Examples

You can run the provided examples to see how the package works:

```bash
python examples/MinimalChannelFlow.py
python examples/LidDrivenCavity.py
python examples/FlowOverAirfoil.py
```

### Project Structure

The project is organized as follows:

```
flowINN/
├── examples/                # Example scripts for different simulations
├── scripts/                 # Utility scripts for cleaning dependencies
├── flowinn/                 # Source code for the package
│   ├── config.py            # Configuration settings
│   ├── mesh/                # Mesh generation and handling
│   ├── models/              # Neural network models
│   ├── physics/             # Physics-based loss functions and boundary conditions
│   ├── plot/                # Plotting and post-processing
│   ├── tests/               # Test cases for different flow problems
│   └── training/            # Training routines and loss functions
├── requirements.txt         # List of dependencies
├── setup.py                 # Setup script for packaging
└── README.md                # Project documentation
```

### Examples

#### Minimal Channel Flow

This example demonstrates a 3D channel flow simulation.

```python
python examples/MinimalChannelFlow.py
```

#### Lid-Driven Cavity

This example demonstrates a 2D lid-driven cavity simulation.

```python
python examples/LidDrivenCavity.py
```

#### Flow Over Airfoil

This example demonstrates a 2D flow over an airfoil simulation.

```python
python examples/FlowOverAirfoil.py
```

### Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
