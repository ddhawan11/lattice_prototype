#from lattice_copy import Lattice
from lattice import Lattice
import numpy as np
###  1D chain

unit_cell = [[1]]
L = [5]
lattice_1d = Lattice(L=L, unit_cell=unit_cell)
lattice_1d.plot_lattice()

### Square Lattice
print("Square")
unit_cell = [[1,0], [0, 1]]
basis = [[0,0]]
        
L = [3,3]
lattice_square = Lattice(L=L, unit_cell=unit_cell, basis=basis, neighbor_order=1)#, pbc=[True, True])

lattice_square.add_edge((0,4,0))
lattice_square.plot_lattice()


### Rectangular Lattice
print("Rectangular")
unit_cell = [[1, 0], [0, 1]]
basis = [[0, 0]]
L = [3, 4]

lattice_rec = Lattice(L=L, unit_cell=unit_cell)
#print(lattice_rec.lattice_points)
lattice_rec.plot_lattice()

### Triangular Lattice
print("Triangular")
unit_cell = [[1, 0],[0.5, np.sqrt(3)/2]]
basis = None
L = [3,3]
lattice_triangle = Lattice(L=L, unit_cell=unit_cell, basis=basis)
#print(lattice_triangle.lattice_points)
lattice_triangle.plot_lattice()


### Honeycomb Lattice
print("Honeycomb")
unit_cell = [[1, 0], [0.5, np.sqrt(3)/2]]
basis = [[0.5, 0.5 / 3**0.5], [1, 1 / 3**0.5]]
L = [2,2]
lattice_honeycomb = Lattice(L=L, unit_cell=unit_cell, basis=basis, pbc=False)
lattice_honeycomb.plot_lattice()


### Lattice using custom edges
unit_cell = [[1, 0], [0.5, 0.75**0.5]]
basis = [[0.5, 0.5 / 3**0.5], [1, 1 / 3**0.5]]
L = [2,2]
custom_edges=[
    (0, 1, [0.5, 0.5 / 3**0.5]),
    (0, 1, [-0.5, 0.5 / 3**0.5]),
    (0, 1, [0, -1 / 3**0.5]),
]
lattice_custom = Lattice(
        unit_cell=[[1, 0], [0.5, 0.75**0.5]],
        L = L,
        basis=[[0.5, 0.5 / 3**0.5], [1, 1 / 3**0.5]],
        custom_edges = custom_edges)

lattice_custom.plot_lattice()