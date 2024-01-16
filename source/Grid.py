#Grid.py

import numpy as np
from source.uservariables import *
from source.Derivatives import *

# For description of the grid setup see https://github.com/GRChombo/engrenage/wiki/Useful-code-background

# hard code number of ghosts to 3 here
num_ghosts = 3

class Grid :
    
    """
    Represents the grid used in the evolution of the
    
    attributes: the size of the grid (max_r)
                number of points num_points_r
                the type of grid (logarithmic or fixed intervals - is_log_grid),
                the grid spacing at the origin base_dx
                the log_factor between intervals in dr
                
    methods: calculates the derivative matrices for the specified grid, 
             and the vector of r values etc
    
    """
    
    # constructor function
    def __init__(self, a_max_r, a_num_points_r, a_log_factor=1.0):

        self.max_r = a_max_r
        self.num_points_r = a_num_points_r
        self.log_factor = a_log_factor
        if self.log_factor == 1.0 :
            self.base_dx = self.max_r / (self.num_points_r - 3.0 - 0.5)
        else :
            N = self.num_points_r - 4
            self.base_dx = self.max_r / ( (self.log_factor ** (N+1.0) - 1.0) /
                                          (self.log_factor - 1.0)             - 0.5)

        # Define the vector of the r values and the intervals between them
        self.r_vector = np.zeros(self.num_points_r)
        self.dr_vector = np.zeros(self.num_points_r)
        
        # Fill with values according to logarithmic scheme (note that fixed spacing is 
        # treated simply as a special case with c=1.0
        self.dr_vector[num_ghosts] = self.base_dx
        self.dr_vector[num_ghosts-1] = self.dr_vector[num_ghosts]/self.log_factor
        self.dr_vector[num_ghosts-2] = self.dr_vector[num_ghosts-1]/self.log_factor
        self.dr_vector[num_ghosts-3] = self.dr_vector[num_ghosts-2]/self.log_factor      
        self.r_vector[num_ghosts] = self.base_dx / 2.0
        self.r_vector[num_ghosts - 1] = - self.base_dx / 2.0
        self.r_vector[num_ghosts - 2] = (self.r_vector[num_ghosts - 1] 
                                         - self.base_dx / self.log_factor)
        self.r_vector[num_ghosts - 3] = (self.r_vector[num_ghosts - 2] 
                                         - self.base_dx / self.log_factor / self.log_factor)
        for idx in np.arange(num_ghosts, a_num_points_r, 1) :
            self.dr_vector[idx] = self.dr_vector[idx-1] * self.log_factor
            self.r_vector[idx] = self.r_vector[idx-1] + self.dr_vector[idx]
        
        # For debugging
        #print("The grid is  ", self.r_vector)
        #print("The spacing is  ", self.dr_vector)
        
        self.derivatives = Derivatives(self.r_vector, self.dr_vector)

    # fills the inner boundary ghosts at r=0 end
    def fill_inner_boundary(self, state) :
    
        for ivar in range(0, NUM_VARS) :
            fill_inner_boundary_ivar(state)
                
    def fill_inner_boundary_ivar(self, state) :
        
        #FIXME with proper interpolation of data
        
        var_parity = parity[ivar]
        c = self.log_factor
        dx = self.base_dx
        N = self.num_points_r
        dist1 = dx / c - c * dx # distance to ghost element -2
        dist2 = dx / c + dx / c / c - c * dx # distance to ghost element -3
        oneoverlogdr_a = 1.0 / (dx * c)
        oneoverlogdr2_a = oneoverlogdr_a * oneoverlogdr_a        
        idx_a = ivar * N + num_ghosts + 1 # Point a is the second valid point in the grid above r=0
        # first impose the symmetry about zero for ghost element -1
        state[idx_a - 2] = state[idx_a - 1] * var_parity
        # calculate gradients at a
        oneplusc = 1.0 + c
        oneplusc2 = 1.0 + c*c
        onepluscplusc2 = 1.0 + c + c*c
        Ap2 = - 1.0 / ( c*c * oneplusc * oneplusc2 * onepluscplusc2 )
        Ap1 = oneplusc / (c*c * onepluscplusc2 )
        A0 = 2.0 * (c - 1.0) / c
        Am1 = - c**4 * Ap1
        Am2 = - c**8 * Ap2
        
        dfdx_a   = (Am2 * state[idx_a-2] + Am1 * state[idx_a-1] + A0 * state[idx_a] 
              + Ap1 * state[idx_a+1] + Ap2 * state[idx_a+2]) * oneoverlogdr_a
        # Use taylor series approximation to fill points
        state[idx_a - 3] = (state[idx_a] + dist1 * dfdx_a) * var_parity
        state[idx_a - 4] = (state[idx_a] + dist2 * dfdx_a) * var_parity            

    # fills the outer boundary ghosts at large r
    def fill_outer_boundary(self, state) :

        for ivar in range(0, NUM_VARS) :
            fill_outer_boundary_ivar(state)
    
    def fill_outer_boundary_ivar(self, state) :
        
        #FIXME with a proper extrapolation
       
        boundary_cells = np.array([(ivar + 1)*N-3, (ivar + 1)*N-2, (ivar + 1)*N-1])
        for count, ix in enumerate(boundary_cells) :
            offset = -1 - count
            state[ix]    = state[ix + offset]