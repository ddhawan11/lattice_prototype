# Unit cell distribution logic
from typing import Union
from collections.abc import Sequence
from textwrap import dedent
import jax
import numpy as np

def comparable(
    x, bin_density=3326400, offset= float(5413/ 15629)):
    """
    Casts a floating point input to integer-indexed bins that are safe to compare
    or hash.

    Arguments:
        x: the float array to be converted
        bin_density: the inverse width of each bin. When binning rational numbers,
            it's best to use a multiple of all expected denominators `bin_density`.
            The default is :math:`3326400 = 2^6\times 3^3\times 5^2\times 7\times 11`.
        offset: constant offset added to `bin_density * x` before rounding. To minimise
            the chances of "interesting" numbers appearing on bin boundaries, it's
            best to use a rational number with a large prime denominator.
            The default is 5413/15629, both are primes.

    Returns:
        `x * bin_density + offset` rounded to an integer

    Example:

        >>> comparable([0.0, 0.3, 0.30000001, 1.3])
        array([      0,  997920,  997920, 4324320])
    """
    return np.asarray(np.rint(np.asarray(x) * bin_density + offset), dtype=int)


def site_to_idx(coords, L, basis):
    """Converts unit cell + sublattice coordinates into lattice site indices."""
    if isinstance(coords, tuple):
        basis_coords, sl = coords
    else:
        basis_coords, sl = coords[:, :-1], coords[:, -1]
    
    # Accepts extended shells (For PBC)
    basis_coords = basis_coords % L
    
    # Index difference between sites one lattice site apart in each direction 
    # len(site_offsets) for the last axis, as all sites in one cell are listed
    # factor of extent[-1] to the penultimate axis, etc.                                                                                              
    radix = np.cumprod([len(basis), *L[:0:-1]])[::-1]
    return basis_coords @ radix + sl


Array = Union[np.ndarray, jax.Array]
JaxArray = jax.Array

CustomEdgeT = Union[tuple[int, int, Array], tuple[int, int, Array, int]]


def get_custom_edges(
    basis_vectors, extent, site_offsets, pbc, atol, custom_edges: Sequence[CustomEdgeT]
):
    """Generates the edges described in `custom_edges` for all unit cells.

    See the docstring of `Lattice.__init__` for the syntax of `custom_edges."""
    if not all([len(desc) in (3, 4) for desc in custom_edges]):
        raise ValueError(
            dedent(
                """
            custom_edges must be a list of tuples of length 3 or 4.
            Every tuple must contain two sublattice indices (integers), a distance vector
            and can optionally include an integer to represent the color of that edge.
            """
            )
        )

    def translated_edges(sl1, sl2, distance, color):
        # get distance in terms of unit cells
        d_cell = (distance + site_offsets[sl1] - site_offsets[sl2]) @ np.linalg.inv(
            basis_vectors
        )

        if not np.all(np.isclose(d_cell, np.rint(d_cell), rtol=0.0, atol=atol)):
            # error out
            msg = f"{distance} is invalid distance vector between sublattices {sl1}->{sl2}"
            # see if the user flipped the vector accidentally
            d_cell = (distance + site_offsets[sl2] - site_offsets[sl1]) @ np.linalg.inv(
                basis_vectors
            )
            if np.all(np.isclose(d_cell, np.rint(d_cell), rtol=0.0, atol=atol)):
                msg += f" (but valid {sl2}->{sl1})"
            raise ValueError(msg)

        d_cell = np.asarray(np.rint(d_cell), dtype=int)

        # catches self-referential and other unrealisable long edges
        if not np.all(d_cell < extent):
            raise ValueError(
                f"Distance vector {distance} does not fit into the lattice"
            )

        # Unit cells of starting points
        start_min = np.where(pbc, 0, np.maximum(0, -d_cell))
        start_max = np.where(pbc, extent, extent - np.maximum(0, d_cell))
        start_ranges = [slice(lo, hi) for lo, hi in zip(start_min, start_max)]
        start = np.mgrid[start_ranges].reshape(len(extent), -1).T
        end = (start + d_cell) % extent

        # Convert to site indices
        print(start, sl1, extent, site_offsets)
        start = site_to_idx((start, sl1), extent, site_offsets)
        end = site_to_idx((end, sl2), extent, site_offsets)

        return [(*edge, color) for edge in zip(start, end)]

    colored_edges = []
    for i, desc in enumerate(custom_edges):
        edge_data = desc[:3]
        edge_color = desc[3] if len(desc) == 4 else i
        colored_edges += translated_edges(*edge_data, edge_color)
    return colored_edges

