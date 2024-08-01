import pennylane as qml
import numpy as np
import math
from lattice import Lattice
from pennylane import X, Y, Z
from pennylane.fermi import FermiWord


def generate_lattice(num_spins, lattice_dim):
    r"""Generates the lattice for Hamiltonian functions"""

    if lattice_dim not in ["1D", "2D"]:
            raise ValueError("Currently only 1D and 2D systems are supported")

    if lattice_dim == "1D":
        L = [num_spins]
        unit_cell = [[1]]
        lattice = Lattice(L=L, unit_cell=unit_cell)
        
    elif lattice_dim == "2D":
        length = int(math.sqrt(num_spins))
        if length**2 == num_spins:
            L = [length, length]
            unit_cell = [[1,0], [0, 1]]
            lattice = Lattice(L=L, unit_cell=unit_cell)
        else:
            raise ValueError(f"{num_spins} is not a perfect square. Please provide a perfect square number to represent the number of spins in a square.")

    return lattice
    
def TFIsingModel(num_spins, coupling, magnetic_field, lattice_dim="1D"):
    r"""Generates the transverse field Ising model on a lattice.
    The Hamiltonian is represented as:
    .. math::

        \hat{H} =  -J \sum_{<i,j>} \sigma_i^{z} \sigma_j^{z} - h\sum{i} \sigma_{i}^{x}

    where J is the coupling defined for the Hamiltonian, h is the strength of transverse
    magnetic field and i,j represent the indices for neighboring spins.
    """

    if not isinstance(num_spins, int) or num_spins <=0:
        raise TypeError("Number of spins must be a positive integer.")

    lattice = generate_lattice(num_spins, lattice_dim)

    hamiltonian = 0.0

    for edge in lattice.edges:
        i, j = edge[0], edge[1]
        hamiltonian += - coupling * (Z(i) @ Z(j))

    for vertex in range(lattice.vertices):
        hamiltonian += -magnetic_field * X(vertex)

    return hamiltonian

def HeisenbergModel(num_spins, coupling, lattice_dim="1D"):
    r"""Generates the Heisenberg model on a lattice.
    The Hamiltonian is represented as:
    .. math::

         \hat{H} = J\sum_{<i,j>}(\sigma_i^x\sigma_j^x + \sigma_i^y\sigma_j^y + \sigma_i^z\sigma_j^z)

    where J is the coupling constant defined for the Hamiltonian, and i,j represent the indices for neighboring spins.
    """

    if not isinstance(num_spins, int) or num_spins <=0:
        raise TypeError("Number of spins must be a positive integer.")

    lattice = generate_lattice(num_spins, lattice_dim)

    hamiltonian = 0.0
    for edge in lattice.edges:
        i, j = edge[0], edge[1]
        hamiltonian += coupling * (X(i) @ X(j) + Y(i) @ Y(j) + Z(i) @ Z(j))

    return hamiltonian

def HubbardModel_spinless(num_spins, t=None, U=1.0, lattice_dim="1D", mapping="jordan_wigner"):
    r"""Generates the Hubbard model on a lattice.
    The Hamiltonian is represented as:
    .. math::

        \hat{H} = -t\sum_{<i,j>, \sigma}(c_{i\sigma}^{\dagger}c_{j\sigma}) + U\sum_{i}n_{i \uparrow} n_{i\downarrow}

    where t is the hopping term representing the kinetic energy of electrons, and U is the on-site Coulomb interaction, representing the repulsion between electrons.
    """

    if not isinstance(num_spins, int) or num_spins <=0:
        raise TypeError("Number of spins must be a positive integer.")

    lattice = generate_lattice(num_spins, lattice_dim)

    hamiltonian = 0.0
    for edge in lattice.edges:
        i, j = edge[0], edge[1]
        hopping_term = t * (FermiWord({(0, i): "+", (1, j):"-"}) + FermiWord({(0, j): "+", (1, i): "-"}))
        int_term = U * FermiWord({(0, i): "+", (1, i):"-", (2, j): "+", (3, j): "-"})
        hamiltonian += hopping_term + int_term

    qubit_ham = qml.qchem.qubit_observable(hamiltonian, mapping=mapping)

    return qubit_ham.simplify()
    



