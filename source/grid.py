import numpy as np

from source.derivatives import Derivatives
from source.spacing import Spacing
from source.uservariables import *


# For description of the grid setup see https://github.com/GRChombo/engrenage/wiki/Useful-code-background


ALL_INDICES = np.arange(NUM_VARS, dtype=np.uint8)


class Grid:
    """Represents the grid used in the evolution of the state."""

    def __init__(self, spacing: Spacing):
        self.r, self.dr_dx = spacing[[0, 1]]
        self.N = self.r.size
        self.dr = spacing.dx * self.dr_dx
        self.der = Derivatives(spacing)
        self.extent = spacing.extent
        self.num_points = self.r.size
        self.min_dr = spacing.min_dr

    def fill_boundaries(self, state, indices=ALL_INDICES):
        self.fill_inner_boundary(state, indices)
        self.fill_outer_boundary(state, indices)

    def fill_inner_boundary(self, state, indices=ALL_INDICES):
        """Fill the inner boundary of the grid.

        There are two possibilities, whether the spacing is full extent or not.
        (non-negative r or r from -r_max to r_max)
        """
        if self.extent == SpacingExtent.HALF:
            # If the r coordinate is positive, fill inner boundary with parity.
            state[indices, :NUM_GHOSTS] = (
                PARITY[indices, None]
                * state[indices, 2 * NUM_GHOSTS - 1: NUM_GHOSTS - 1: -1]
            )
        else:
            # If the r coordinate goes from -r_max to r_max, fill the inner boundary like the outer one.
            self.fill_outer_boundary(state[..., ::-1], indices)

    def fill_outer_boundary(self, state, indices=ALL_INDICES):
        # For outer boundaries, we assume a law of the form: a + b * r**n
        # "a" is ASYMP_OFFSET and "n" is ASYMP_POWER, "b" is to be determined
        # on last point before ghost points.
        idx = -NUM_GHOSTS - 1
        outer_state = state[:, -NUM_GHOSTS:]
        b = (state[indices, idx] - ASYMP_OFFSET[indices]) / self.r[idx] ** ASYMP_POWER[indices]

        outer_state[indices, -NUM_GHOSTS:] = (
            ASYMP_OFFSET[indices, None]
            + b[..., None] * self.r[-NUM_GHOSTS:] ** ASYMP_POWER[indices, None]
        )

    def get_first_derivative(self, array: np.ndarray, indices=None):
        """Compute the first derivative of an array for the specified indices."""
        dr_array = np.zeros_like(array)
        dr_array[indices] = array[indices] @ self.der.drn_matrix[1].T
        return dr_array / self.dr

    def get_second_derivative(self, array: np.ndarray, indices=None):
        """Compute the second derivative of an array for the specified indices."""
        dr2_array = np.zeros_like(array)
        dr2_array[indices] = array[indices] @ self.der.drn_matrix[2].T
        return dr2_array / self.dr**2

    def get_advection(self, array: np.ndarray, direction: np.ndarray, indices=None):
        """Compute the advection of an array along a direction for the specified indices."""

        # Direction of advection is given by direction array.
        # True or 1 is right advection and False or 0 is left advection.
        advec_matrix = self.der.advec_x_matrix[direction.astype(int), np.arange(direction.size)]
        advec_array = np.zeros_like(array)
        advec_array[indices] = array[indices] @ advec_matrix.T / self.dr
        return advec_array

    def get_kreiss_oliger_diss(self, state: np.ndarray, indices=None):
        # Compute the second derivative of the ivars in argument
        diss_state = np.zeros_like(state)
        diss_state[indices] = state[indices] @ self.der.drn_matrix[6].T
        return diss_state / (2 ** 6 * self.dr)
