#hamdiagnostic.py

# python modules
import numpy as np

# homemade code
from core.grid import *
from bssn.tensoralgebra import *

# The diagnostic function returns the Hamiltonian constraint over the grid
# it takes in the solution of the evolution, which is the state vector at every
# time step, and returns the spatial profile Ham(r) at each time step
def get_Ham_diagnostic(states_over_time, t, grid: Grid, background, matter) :
    
    # For readability
    r = grid.r
    N = grid.num_points
    num_times = int(np.size(states_over_time) / (grid.NUM_VARS * N))
    Ham = np.zeros([num_times, N])
    
    # unpack the vectors at each time
    for i in range(num_times) :
        
        if(num_times == 1):
            state = states_over_time
            t_i = t
        else :
            state = states_over_time[i]
            t_i = t[i]

        state = state.reshape(grid.NUM_VARS, -1)

        # Assign the variables to parts of the solution
        N = grid.N
        bssn_vars = BSSNVars(N)
        bssn_vars.set_bssn_vars(state)
        matter.set_matter_vars(state, bssn_vars, grid)
    
        # get the derivatives of the bssn vars in tensor form - see bssnvars.py
        d1 = grid.get_d1_metric_quantities(state)
        d2 = grid.get_d2_metric_quantities(state)       
        
        # Calculate some useful quantities
        ########################################################
        
        em4phi = np.exp(-4.0*bssn_vars.phi)

        # (unscaled) \bar\gamma_ij and \bar\gamma^ij
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
        bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)

        # The connections Delta^i, Delta^i_jk and Delta_ijk
        Delta_U, Delta_ULL, Delta_LLL  = get_tensor_connections(r, bssn_vars.h_LL, d1.h_LL, background)    
        bar_chris = get_bar_christoffel(r, Delta_ULL, background)  
        bar_Rij = get_bar_ricci_tensor(r, bssn_vars.h_LL, d1.h_LL, d2.h_LL, bssn_vars.lambda_U, d1.lambda_U, 
                         Delta_U, Delta_ULL, Delta_LLL, 
                         bar_gamma_UU, bar_gamma_LL, background)
        bar_R   = get_trace(bar_Rij, bar_gamma_UU)
        
        # Asquared = \bar A_ij \bar A^ij
        Asquared = get_bar_A_squared(r, bssn_vars, background)        
        
        # Matter sources
        my_emtensor = matter.get_emtensor(r, bssn_vars, background)

        # End of: Calculate some useful quantities, now start diagnostic
        #################################################################

        # Get the Ham constraint eqn (13) of Baumgarte https://arxiv.org/abs/1211.6632
        Ham[i,:] = (  two_thirds * bssn_vars.K * bssn_vars.K - Asquared
                      + em4phi * ( bar_R
                                   - 8.0 * np.einsum('xij,xi,xj->x', bar_gamma_UU, d1.phi, d1.phi)
                                   - 8.0 * np.einsum('xij,xij->x', bar_gamma_UU, d2.phi)
                                   + 8.0 * np.einsum('xij,xkij,xk->x', bar_gamma_UU, bar_chris, d1.phi))
                         - 2.0 * eight_pi_G * my_emtensor.rho)

        # Fix endpoints
        grid.fill_inner_boundary_single_variable(Ham[i,:])
    
    
    # end of iteration over time  
    #########################################################################
    
    return Ham
