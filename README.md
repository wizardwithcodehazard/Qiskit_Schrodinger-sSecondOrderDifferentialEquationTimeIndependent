# Schrödinger's Time-Independent Equation Solver
documentation : https://carnation-jackfruit-cd3.notion.site/Solving-Schr-dinger-s-Time-Independent-Equation-Using-Quantum-Computing-91bb20d4666841be88c867c9278b8fdd
This repository contains a Python implementation to solve Schrödinger's time-independent equation using both classical and quantum computing methods. The classical approach uses the finite difference method to construct the Hamiltonian matrix, while the quantum approach utilizes Qiskit for solving eigenvalue problems.

## Table of Contents

- [Overview](#overview)
- [Theoretical Background](#theoretical-background)
- [Code Explanation](#code-explanation)
  - [Classical Approach](#classical-approach)
  - [Quantum Approach](#quantum-approach)
- [Assumptions](#assumptions)
- [Installation and Dependencies](#installation-and-dependencies)
- [License](#license)

## Overview

Schrödinger's time-independent equation is a fundamental equation in quantum mechanics that describes how the quantum state of a physical system changes over time. This repository includes two approaches to solving the equation:

1. **Classical Approach**: Uses the finite difference method to discretize the equation and solve it using standard linear algebra techniques.
2. **Quantum Approach**: Uses Qiskit, a quantum computing framework, to solve the eigenvalue problem using the Variational Quantum Eigensolver (VQE) algorithm.

## Theoretical Background

### Schrödinger's Time-Independent Equation

The time-independent Schrödinger equation describes the quantum state of a physical system where the Hamiltonian operator \(H\) and the energy eigenvalue \(E\) are used to find the wave function \(\psi(x)\).

### Finite Difference Method

To approximate the second-order derivative in the Schrödinger equation, the finite difference method is used. The Hamiltonian matrix \(H\) for a particle in a one-dimensional potential well is constructed based on the discretized domain and the finite difference approximation of the second derivative.

## Code Explanation

### Classical Approach

The classical approach involves the following steps:

1. **Discretization**: Discretize the domain into grid points.
2. **Hamiltonian Construction**: Construct the Hamiltonian matrix using the finite difference method.
3. **Eigenvalue Calculation**: Compute the eigenvalues and eigenfunctions using the `scipy.linalg.eigh` function.

```python
import numpy as np
from scipy.linalg import eigh

# Parameters
L = 1.0
N = 4
hbar = 1.0
m = 1.0

# Discretize the domain
x = np.linspace(0, L, N)
dx = x[1] - x[0]

# Define the potential function
V = np.zeros(N)

# Construct the Hamiltonian matrix
H = -hbar**2 / (2 * m * dx**2) * (np.diag(-2 * np.ones(N)) +
                                  np.diag(np.ones(N-1), 1) +
                                  np.diag(np.ones(N-1), -1)) + np.diag(V)

# Compute eigenvalues
eigenvalues, eigenvectors = eigh(H)
```

### Quantum Approach

The quantum approach involves using Qiskit to solve the eigenvalue problem:

1. **Hamiltonian Conversion**: Convert the Hamiltonian matrix to a format suitable for Qiskit.
2. **VQE Algorithm**: Apply the Variational Quantum Eigensolver (VQE) to compute the eigenvalues.

```python
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import Estimator
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.eigensolvers import NumPyEigensolver

# Convert Hamiltonian to Qiskit format
matrix_qiskit = SparsePauliOp.from_operator(H)

# Function to compute eigenvalues using VQE
def compute_eigenvalue_vqe(matrix):
    ansatz = EfficientSU2(num_qubits=N, entanglement='linear', reps=2)
    optimizer = COBYLA(maxiter=1000)
    estimator = Estimator()
    vqe = VQE(ansatz=ansatz, optimizer=optimizer, estimator=estimator)
    result = vqe.compute_minimum_eigenvalue(matrix)
    return result.eigenvalue

# Compute eigenvalues using VQE
vqe_eigenvalues = [compute_eigenvalue_vqe(matrix_qiskit) for _ in range(num_states)]
```

## Assumptions

- The potential well is assumed to be an infinite square well where \(V(x) = 0\) within the well.
- The domain is discretized uniformly, and the grid points are evenly spaced.
- For the quantum approach, the number of qubits \(N\) should be chosen such that it is feasible for the chosen ansatz and optimizer.

## Installation and Dependencies

To run the code, you will need the following Python packages:

- `numpy`
- `scipy`
- `matplotlib`
- `qiskit`
- `qiskit_algorithms`

You can install these packages using pip:

```bash
pip install numpy scipy matplotlib qiskit qiskit_algorithms
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

