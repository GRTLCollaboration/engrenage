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
        self.calculate_interpolation_weights()

    # fills the inner boundary ghosts at r=0 end
    def fill_inner_boundary(self, state) :
    
        for ivar in range(0, NUM_VARS) :
           self.fill_inner_boundary_ivar(state, ivar)
                
    def fill_inner_boundary_ivar(self, state, ivar) :
        
        var_parity = parity[ivar]
        N = self.num_points_r
       
        # idx1 is the first valid point in the grid above r=0
        idx_1 = ivar * N + num_ghosts 
        state[idx_1 - 1] = state[idx_1] * var_parity
        state[idx_1 - 2] = var_parity * (state[idx_1+0] * self.xA_weights[0] + 
                                         state[idx_1+1] * self.xA_weights[1] +
                                         state[idx_1+2] * self.xA_weights[2] +
                                         state[idx_1+3] * self.xA_weights[3])
        state[idx_1 - 3] = var_parity * (state[idx_1+0] * self.xB_weights[0] + 
                                         state[idx_1+1] * self.xB_weights[1] +
                                         state[idx_1+2] * self.xB_weights[2] + 
                                         state[idx_1+3] * self.xB_weights[3])
        
    # fills the outer boundary ghosts at large r
    def fill_outer_boundary(self, state) :

        for ivar in range(0, NUM_VARS) :
            self.fill_outer_boundary_ivar(state, ivar)
    
    def fill_outer_boundary_ivar(self, state, ivar) :
        
        var_asymptotic_power = asymptotic_power[ivar]
        N = self.num_points_r
        offset = 0
        if (ivar == idx_lapse) :
            offset = 1.0
       
        # idx1 is the last valid point in the grid below ghosts for large r
        idx_1 = (ivar+1) * N - num_ghosts - 1 
        AA = (state[idx_1 + 0] - offset) / (self.r_vector[N-num_ghosts-1] ** var_asymptotic_power)
        state[idx_1 + 1] = AA * (self.r_vector[N-num_ghosts+0] ** var_asymptotic_power) + offset
        state[idx_1 + 2] = AA * (self.r_vector[N-num_ghosts+1] ** var_asymptotic_power) + offset
        state[idx_1 + 3] = AA * (self.r_vector[N-num_ghosts+2] ** var_asymptotic_power) + offset
            
    def calculate_interpolation_weights(self) :
        
        # For readability
        c = self.log_factor
        c2 = c*c
        c3 = c2 * c
        c4 = c2 * c2
        c5 = c2 * c3
        c6 = c3 * c3
        c7 = c3 * c4
        c8 = c4 * c4
        c9 = c5 * c4
        oneplusc = 1.0 + c
        oneplusc2 = 1.0 + c*c
        onepluscplusc2 = 1.0 + c + c*c
        
        # Weights for interpolation at the end points, using the innermost valid
        # 4 grid points above r=0, label the values at these f1, f2, f3, f4
        # WA is for the point at r = -dx/2 - dx/c (ignore parity)

        WA1 = (c8 + c7 - 2*c5 - 2*c4 + 2*c2 + c - 1)/ (c6 * onepluscplusc2)
        WA2 = (c6 + c5 + c4 - c3 - c2 - c + 1)/ c8
        WA3 = (-c5 + c2 + c - 1) / c9
        WA4 = (c4 - c2 - c + 1) / (c9 * onepluscplusc2)

        self.xA_weights = np.array([WA1, WA2, WA3, WA4])

        # WB is the point at r = -dx/2 - dx/c - dx/c/c (ignore parity)
        WB1 = (c9 - c7 - 2*c6 - c5 + c4 +3*c3 + c2 - c - 1) / c9
        WB2 = (c3 * oneplusc - c - 1) * (c3 * onepluscplusc2 - oneplusc) / c9 / c2
        WB3 = (-c3 + oneplusc) * (c3 * onepluscplusc2 - oneplusc) / c9 / c3
        WB4 = (c5 - 2*c3 - c2 + c + 1) / c9 / c3
        
        self.xB_weights = np.array([WB1, WB2, WB3, WB4])