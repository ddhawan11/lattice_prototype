import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.spatial import cKDTree
import utils

class Lattice:
    def __init__(self, L, unit_cell, basis=None, pbc=False, neighbor_order=None, custom_edges= None, distance_atol=1e-5):
        """
        L: Number of unit cells in a direction, it is a list depending on the dimensions of the lattice
        unit_cell: unit cell for a lattice, differs for each lattice.
        basis: Initial positions of atoms.
        pbc: Boundary conditions, boolean or series of bools depending on the lattice dimensions.
        distance_atol: Distance below which spatial points are considered equal for the purpose of identifying nearest neighbors. 
        """

        self.L = np.asarray(L)
        self.n_dim = len(L)
        self.unit_cell = np.asarray(unit_cell)
        if basis is None:
            basis = np.zeros(self.unit_cell.shape[0])[None, :]
        self.basis = np.asarray(basis)
        self.pbc = pbc
        self.test_input_accuracy()

        if self.pbc:
            extra_shells = np.where(self.pbc, neighbor_order, 0)
        else:
            extra_shells = None

        self.coords, self.lattice_points = self.generate_grid(extra_shells)

        if custom_edges is None:
            if neighbor_order is None:
                neighbor_order = 1
            cutoff = neighbor_order * np.linalg.norm(self.unit_cell, axis=1).max() + distance_atol
            self.generate_edges(cutoff, neighbor_order)
            self.generate_colored_edges(neighbor_order)
        else:
            if neighbor_order is not None:
                raise ValueError(
                    "custom_edges and neighbor_order cannot be specified at the same time"
                )
            self.edges = utils.get_custom_edges(
                self.unit_cell,
                self.L,
                self.basis,
                self.pbc,
                distance_atol,
                custom_edges,
            )

    def test_input_accuracy(self):
        for l in self.L:
            if (not isinstance(l, np.int64)) or l <= 0:
                raise TypeError("Argument `L` must be a positive integer")

        if self.unit_cell.ndim != 2:
            raise ValueError(
                "'unit_cell' must have ndim==2 (as array of primitive vectors)"
            )
        if self.unit_cell.shape[0] != self.unit_cell.shape[1]:
            raise ValueError("The number of primitive vectors must match their length")
        ## Need to check if all the positions in basis are unique and also if the pbc is defined correctly.
               
    def generate_edges(self, cutoff, neighbor_order):
        self.tree = cKDTree(self.lattice_points)  # Create the KD-tree
        self.edges = self.identify_neighbors(cutoff, neighbor_order)

    def identify_neighbors(self, cutoff, neighbor_order):
        # Use tree.query to find neighbors within the cutoff distance
        # We use query_ball_tree to find all points within the cutoff distance
        indices = self.tree.query_ball_tree(self.tree, cutoff)  # Returns a list of indices for neighbors

        unique_pairs = set()
        row = []
        col = []
        distance=[]
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors:
                if neighbor != i:
                    pair = (min(i, neighbor), max(i, neighbor))
                    if pair not in unique_pairs:  # Check if the pair is already added
                        unique_pairs.add(pair)  # Add pair to the set
                        # Calculate the distance
                        dist = np.linalg.norm(self.lattice_points[i] - self.lattice_points[neighbor])
                        row.append(i)
                        col.append(neighbor)
                        distance.append(dist)

        row = np.array(row)
        col = np.array(col)
        distance = utils.comparable(np.array(distance))
        _, ii = np.unique(distance, return_inverse=True)

        # Getting sorted pairs based on unique distances
        results = [sorted(list(zip(row[ii == k], col[ii == k]))) for k in range(neighbor_order)]
        return results

    def generate_colored_edges(self, neighbor_order):
        ids = utils.site_to_idx(self.coords, self.L, self.basis)
        colored_edges = []
        for k, naive_edges in enumerate(self.edges):
            true_edges = set()
            for node1, node2 in naive_edges:
                # switch to real node indices
                node1 = ids[node1]
                node2 = ids[node2]
                if node1 == node2:
                    raise RuntimeError(
                        f"Lattice contains self-referential edge {(node1, node2)} of order {k}"
                    )
                elif node1 > node2:
                    node1, node2 = node2, node1
                true_edges.add((node1, node2))
            for edge in true_edges:
                colored_edges.append((*edge, k))
        self.edges = colored_edges

    def add_edge(self, edge_index):
        if len(edge_index) == 2:
            edge_index = (*edge_index, 0)

        self.edges.append(edge_index)


    def generate_grid(self, extra_shells):
        """Generates the coordinates of all lattice sites.
        extra_shells: (optional) the number of unit cells added along each lattice direction.
        This is used for near-neighbour searching in periodic BCs. It must be a vector of the
        same length as L"""

        if extra_shells is None:  ## For Periodic boundary conditions, including hidden nodes
            extra_shells = np.zeros(self.L.size, dtype=int)

        shell_min = -extra_shells
        shell_max = self.L + extra_shells
        ranges = [slice(lo, hi) for lo, hi in zip(shell_min, shell_max)]

        # site coordinate within unit cell
        ranges += [slice(0, len(self.basis))]
        basis_coords = np.mgrid[ranges].reshape(len(self.L) + 1, -1).T
        lattice_points = np.matmul(basis_coords[:, :-1],self.unit_cell)
        lattice_points = lattice_points.reshape(-1, len(self.basis), len(self.L)) + self.basis
        lattice_points = lattice_points.reshape(-1, len(self.L))
        return basis_coords, lattice_points

    def plot_lattice(self): ### Doesn't work with pbc for now
        """
        Plots edges between points in a 2D space.
        
        :param edges: List of tuples where each tuple represents an edge as (start_index, end_index, color).
    :param positions: Numpy array of shape (N, 2), where N is the number of points and each point is in (x, y) format.
    """
        # Create a figure and axis
        plt.figure(figsize=(8, 6))

        # Iterate over the edges and plot each one
        for edge in self.edges:
            # From the edge tuple (start_index, end_index, color)
            start_index, end_index, color = edge
            # Get the starting and ending positions
            start_pos = self.lattice_points[start_index]
            end_pos = self.lattice_points[end_index]

            # Plotting the edge as a line
            x_axis = [start_pos[0], end_pos[0]]
            if self.n_dim == 1:
                y_axis = [0,0]
            else:
                y_axis = [start_pos[1], end_pos[1]]
            plt.plot(x_axis, 
                     y_axis, 
                     color='b')  

        # Scatter the positions to visualize points as well
        if self.n_dim == 1:
            plt.scatter(self.lattice_points, np.zeros_like(self.lattice_points), color='r', s=100)
            for index, pos in enumerate(self.lattice_points):
                plt.text(pos, 0.00, str(index), fontsize=12, ha='center', va='bottom')  

        else:
            plt.scatter(self.lattice_points[:, 0], self.lattice_points[:, 1], color='r', s=100)
            for index, pos in enumerate(self.lattice_points):
                plt.text(pos[0], pos[1] + 0.05, str(index), fontsize=12, ha='center', va='bottom')

        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
            
        plt.grid()
        plt.show()


    

