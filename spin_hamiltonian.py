import pennylane as qml
import numpy as np
from lattice import Lattice
from pennylane import X, Y, Z
from pennylane.fermi import FermiWord


def TFIsingModel(lattice, J=None, h=1.0):
    r"""Generates the transverse field Ising model on a lattice.
    The Hamiltonian is represented as:
    .. math::

        \hat{H} =  -J \sum_{<i,j>} \sigma_i^{z} \sigma_j^{z} - h\sum{i} \sigma_{i}^{x}

    where J is the coupling defined for the Hamiltonian, h is the strength of transverse
    magnetic field and i,j represent the indices for neighboring spins.
    """

    hamiltonian = 0.0
    if J is None:
        if lattice.custom_edges is None:
            raise ValueError("Either J values or custom edges need to be provided.")
        else:
            for edge in lattice.edges:
                i, j = edge[0], edge[1]
                hamiltonian += - edge[2] * (Z(i) @ Z(j))
    else:
        if lattice.custom_edges is None:
            if isinstance(J, (int, float, complex)):
                for edge in lattice.edges:
                    i, j = edge[0], edge[1]
                    hamiltonian += -J * (Z(i) @ Z(j))
            else:
                for edge in lattice.edges:
                    i, j = edge[0], edge[1]
                    hamiltonian += -J[i][j] * (Z(i) @ Z(j))
        else:
            raise ValueError("Both J value and custom edges cannot be provided.")

    for vertex in range(lattice.vertices):
        hamiltonian += -h * X(vertex)

    return hamiltonian

def HeisenbergModel(lattice, J=None):
    r"""Generates the Heisenberg model on a lattice.
    The Hamiltonian is represented as:
    .. math::

         \hat{H} = J\sum_{<i,j>}(\sigma_i^x\sigma_j^x + \sigma_i^y\sigma_j^y + \sigma_i^z\sigma_j^z)

    where J is the coupling constant defined for the Hamiltonian, and i,j represent the indices for neighboring spins.
    """

    hamiltonian = 0.0
    if J is None:
        if lattice.custom_edges is None:
            raise ValueError("Either J values or custom edges need to be provided.")
        else:
            for edge in lattice.edges:
                i, j = edge[0], edge[1]
                hamiltonian += edge[2] * (X(i) @ X(j) + Y(i) @ Y(j) + Z(i) @ Z(j))
    else:
        if lattice.custom_edges is None:
            if isinstance(J, (int, float, complex)):
                for edge in lattice.edges:
                    i, j = edge[0], edge[1]
                    hamiltonian += J * (X(i) @ X(j) + Y(i) @ Y(j) + Z(i) @ Z(j))
            else:
                for edge in lattice.edges:
                    i, j = edge[0], edge[1]
                    hamiltonian += J[i][j] * (X(i) @ X(j) + Y(i) @ Y(j) + Z(i) @ Z(j))
                
        else:
            raise ValueError("Both J value and custom edges cannot be provided.")
    return hamiltonian

def HubbardModel_spinless(lattice, t=None, U=1.0, mapping="jordan_wigner"):
    r"""Generates the Hubbard model on a lattice.
    The Hamiltonian is represented as:
    .. math::

        \hat{H} = -t\sum_{<i,j>, \sigma}(c_{i\sigma}^{\dagger}c_{j\sigma}) + U\sum_{i}n_{i \uparrow} n_{i\downarrow}

    where t is the hopping term representing the kinetic energy of electrons, and U is the on-site Coulomb interaction, representing the repulsion between electrons.
    """

    hamiltonian = 0.0
    if t is None:
        if lattice.custom_edges is None:
            raise ValueError("Either t values or custom edges need to be provided.")
        else:
            for edge in lattice.edges:
                i, j = edge[0], edge[1]
                hopping_term = -edge[2][0] * (FermiWord({(0, i): "+", (1, j):"-"}) + FermiWord({(0, j): "+", (1, i): "-"}))
                int_term = edge[2][1] * FermiWord({(0, i): "+", (1, i):"-", (2, j): "+", (3, j): "-"})
                hamiltonian += hopping_term + int_term
    else:
        if lattice.custom_edges is None:
            if isinstance(t, (int, float, complex)):
                for edge in lattice.edges:
                    i, j = edge[0], edge[1]
                    hopping_term = -t * (FermiWord({(0, i): "+", (1, j):"-"}) + FermiWord({(0, j): "+", (1, i): "-"}))                    
                    int_term = U * FermiWord({(0, i): "+", (1, i):"-", (2, j): "+", (3, j): "-"})
                    hamiltonian += hopping_term + int_term
            else:
                for edge in lattice.edges:
                    i, j = edge[0], edge[1]
                    hopping_term = -t[i][j] * (FermiWord({(0, i): "+", (1, j):"-"}) + FermiWord({(0, j): "+", (1, i): "-"}))                    
                    int_term = U[i][j] * FermiWord({(0, i): "+", (1, i):"-", (2, j): "+", (3, j): "-"})
                    hamiltonian += hopping_term + int_term
                
        else:
            raise ValueError("Both t, U values and custom edges cannot be provided.")
    qubit_ham = qml.qchem.qubit_observable(hamiltonian, mapping=mapping)

    return qubit_ham.simplify()
    



