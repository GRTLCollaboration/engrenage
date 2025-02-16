#hamdiagnostic.py

# python modules
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# homemade code
from core.grid import *
from bssn.tensoralgebra import *

# The diagnostic function returns the expansion over the grid
# it takes in the solution of the evolution, which is the state vector at every
# time step, and returns the spatial profile theta(r) at each time step
# It finds the zero-crossings of such function, which define the apparent
# horizon, and are used to return the mass
def get_horizonfinder(states_over_time, t, grid: Grid, background, matter) :
    
    # For readability
    r = grid.r
    N = grid.num_points
    num_times = int(np.size(states_over_time) / (grid.NUM_VARS * N))
    theta = np.zeros([num_times, N])
    ah_rad = np.zeros([num_times])
    bh_mass = np.zeros([num_times])

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
        
        # Calculate some useful quantities
        ########################################################
        
        em4phi = np.exp(-4.0*bssn_vars.phi)
        ep2phi = np.exp(2.0*bssn_vars.phi)

        # (unscaled) \bar\gamma_ij and \bar\gamma^ij
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
        bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)
        
        # \bar A_ij
        A_LL = get_bar_A_UU(r, bssn_vars, background)    
        
        # End of: Calculate some useful quantities, now start diagnostic
        #################################################################

        # Expansion given by eqn. (49) of NRPy+ https://arxiv.org/abs/1712.07658
        dr_gamma_tt = 2.0 * r + 2.0 * r * bssn_vars.h_LL[:,i_t,i_t] + r**2 * d1.h_LL[:,i_t,i_t,i_r]
        div = ep2phi * bar_gamma_LL[:,i_t,i_t] * np.sqrt(bar_gamma_LL[:,i_r,i_r])
        bar_Ktt = A_LL[:,i_t,i_t] + one_third * bar_gamma_LL[:,i_t,i_t] * bssn_vars.K
        theta[i,:] = (4 * bar_gamma_LL[:,i_t,i_t] * d1.phi[:,i_r] + dr_gamma_tt) / div - 2 * bar_Ktt / bar_gamma_LL[:,i_t,i_t]
        
        # Fix endpoints
        grid.fill_inner_boundary_single_variable(theta[i,:])

        # Find horizon, i.e. zero crossings of theta (r * theta for better convergence)
        theta_i = theta[i,:]
        min_theta = min(theta_i)
        rmin = r[np.where(theta_i==min_theta)][0]
        if min_theta > 0:
            # no horizon
            ah_rad[i] = 0
            bh_mass[i] = 0
        else:
            th_interp = interp1d(r, abs(r) * theta_i)
            gamma_tt_interp = interp1d(r, bar_gamma_LL[:,i_t,i_t] / em4phi)
            gamma_pp_interp = interp1d(r, bar_gamma_LL[:,i_p,i_p] / em4phi)

            # Root finding brentq function
            ah_rad[i] = brentq(th_interp, rmin, r[-1])
                        
            # Area is calculated via the induced metric sqrt(gamma) dtheta dphi
            area = 4.0 * np.pi * np.sqrt(gamma_tt_interp(ah_rad[i]) * gamma_pp_interp(ah_rad[i]))
            bh_mass[i] = np.sqrt(area/(16.0*np.pi))
    
    # end of iteration over time  
    #########################################################################
    
    return theta, ah_rad, bh_mass