import numpy as np

from core.derivatives import Derivatives
from core.spacing import Spacing, NUM_GHOSTS, SpacingExtent
from core.statevector import *
from bssn.bssnvars import *

# coordinates agnostic to coordinate system, assumes x1 is the simulated axis, others are symmetric
i_x1 = 0 # the simulated direction (e.g. r in spherical polar, spherical symmetry)
i_x2 = 1
i_x3 = 2

# For description of the grid setup see https://github.com/GRChombo/engrenage/wiki/Useful-code-background
class Grid:
    """Represents the grid used in the evolution of the state."""

    def __init__(self, spacing: Spacing, a_state_vector : StateVector):
        self.r, self.dr_dx = spacing[[0, 1]]
        self.N = self.r.size
        self.dr = spacing.dx * self.dr_dx
        self.derivs = Derivatives(spacing)
        self.extent = spacing.extent
        self.num_points = self.r.size
        self.min_dr = spacing.min_dr
        self.NUM_VARS = a_state_vector.NUM_VARS
        self.PARITY = a_state_vector.PARITY
        self.ASYMP_OFFSET = a_state_vector.ASYMP_OFFSET
        self.ASYMP_POWER = a_state_vector.ASYMP_POWER
        self.ALL_INDICES = np.arange(self.NUM_VARS, dtype=np.uint8)

    def fill_boundaries(self, state, indices=None):
        
        # Default to all indices if none were specified
        if (indices == None) :
            indices = self.ALL_INDICES
        
        self.fill_inner_boundary(state, indices)
        self.fill_outer_boundary(state, indices)

    def fill_inner_boundary(self, state, indices=None):
        
        """Fill the inner boundary of the grid.
        There are two possibilities, whether the spacing is full extent or not.
        (non-negative r or r from -r_max to r_max)
        """

        # Default to all indices if none were specified
        if (indices == None) :
            indices = self.ALL_INDICES
        
        if self.extent == SpacingExtent.HALF:
            # If the r coordinate is positive, fill inner boundary with parity.
            state[indices, :NUM_GHOSTS] = (
                self.PARITY[indices, None]
                * state[indices, 2 * NUM_GHOSTS - 1: NUM_GHOSTS - 1: -1]
            )
        else:
            # If the r coordinate goes from -r_max to r_max, fill the inner boundary like the outer one.
            self.fill_outer_boundary(state[..., ::-1], indices)

    def fill_inner_boundary_single_variable(self, state, parity=1):
        """Fill the inner boundary of the grid.

        There are two possibilities, whether the spacing is full extent or not.
        (non-negative r or r from -r_max to r_max)
        """
        if self.extent == SpacingExtent.HALF:
            # If the r coordinate is positive, fill inner boundary with parity.
            state[:NUM_GHOSTS] = parity * state[2 * NUM_GHOSTS - 1: NUM_GHOSTS - 1: -1]
        else:
            # If the r coordinate goes from -r_max to r_max, fill the inner boundary like the outer one.
            self.fill_outer_boundary_single_variable(state[::-1])
            
    def fill_outer_boundary(self, state, indices=None):
        # For outer boundaries, we assume a law of the form: a + b * r**n
        # "a" is ASYMP_OFFSET and "n" is ASYMP_POWER, "b" is to be determined
        # on last point before ghost points.

        # Default to all indices if none were specified
        if (indices == None) :
            indices = self.ALL_INDICES        
        
        idx = -NUM_GHOSTS - 1
        outer_state = state[:, -NUM_GHOSTS:]
        b = (state[indices, idx] - self.ASYMP_OFFSET[indices]) / self.r[idx] ** self.ASYMP_POWER[indices]

        outer_state[indices, -NUM_GHOSTS:] = (
            self.ASYMP_OFFSET[indices, None]
            + b[..., None] * self.r[-NUM_GHOSTS:] ** self.ASYMP_POWER[indices, None]
        )

    def fill_outer_boundary_single_variable(self, state, asymp_power=0, asymp_offset=0):
        # For outer boundaries, we assume a law of the form: a + b * r**n
        # "a" is ASYMP_OFFSET and "n" is ASYMP_POWER, "b" is to be determined
        # on last point before ghost points.
        idx = -NUM_GHOSTS - 1
        outer_state = state[:, -NUM_GHOSTS:]
        b = (state[idx] - asymp_offset) / self.r[idx] ** asymp_power

        outer_state[-NUM_GHOSTS:] = asymp_offset + b * self.r[-NUM_GHOSTS:] ** asymp_power
        
    def get_first_derivative(self, array: np.ndarray, indices=None):
        """Compute the first derivative of an array for the specified indices."""
        dr_array = np.zeros_like(array)
        dr_array[indices] = array[indices] @ self.derivs.drn_matrix[1].T
        return dr_array / self.dr

    def get_second_derivative(self, array: np.ndarray, indices=None):
        """Compute the second derivative of an array for the specified indices."""
        dr2_array = np.zeros_like(array)
        dr2_array[indices] = array[indices] @ self.derivs.drn_matrix[2].T
        return dr2_array / self.dr**2

    def get_advection(self, array: np.ndarray, direction: np.ndarray, indices=None):
        """Compute the advection of an array along a direction for the specified indices."""

        # Direction of advection is given by direction array.
        # True or 1 is right advection and False or 0 is left advection.
        advec_matrix = self.derivs.advec_x_matrix[direction.astype(int), np.arange(direction.size)]
        advec_array = np.zeros_like(array)
        advec_array[indices] = array[indices] @ advec_matrix.T / self.dr
        return advec_array

    def get_kreiss_oliger_diss(self, state: np.ndarray, indices=None):
        # Compute the second derivative of the ivars in argument
        diss_state = np.zeros_like(state)
        # Consider trying dxn matrix here to simplify this
        diss_state[indices] = state[indices] @ self.derivs.drn_matrix[6].T
        return diss_state / (2 ** 6 * self.dr)
    
    # This is a helper method that returns the first derivatives needed for BSSN 
    def get_d1_metric_quantities(self, state) :
        
        d1 = BSSNFirstDerivs(self.N)
        
        # first derivatives
        d1_state = self.get_first_derivative(state, d1.first_derivative_indices)
        d1.set_bssn_first_derivs(d1_state)
        
        return d1
    
    # This is a helper method that returns the second derivatives needed for BSSN 
    def get_d2_metric_quantities(self, state) :
        
        d2 = BSSNSecondDerivs(self.N)
        
        # second derivatives
        d2_state = self.get_second_derivative(state, d2.second_derivative_indices)
        d2.set_bssn_second_derivs(d2_state)
        
        return d2
    
    # This is a helper method that returns the second derivatives needed for BSSN 
    def get_advection_d1_metric_quantities(self, state, shift_U) :
        
        advec = BSSNAdvecDerivs(self.N)
        
        # second derivatives
        advec_state = self.get_advection(state, shift_U[:,i_x1] >= 0, advec.advec_indices)
        advec.set_bssn_advec_derivs(advec_state)
        
        return advec
        
