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
# time step, and returns the spatial profile omega(r) at each time step
# It finds the zero-crossings of such function, which define the apparent
# horizon, and are used to return the mass
def get_horizon_diagnostics(states_over_time, t, grid: Grid, background, matter) :
    
    # For readability
    r = grid.r
    N = grid.num_points
    num_times = int(np.size(states_over_time) / (grid.NUM_VARS * N))
    omega = np.zeros([num_times, N])
    ah_radius = np.zeros([num_times])
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
        A_LL = get_bar_A_LL(r, bssn_vars, background)    
        
        # d_r \bar\gamma_tt
        d1_bargamma_tt_dr = 2.0 * r + 2.0 * r * bssn_vars.h_LL[:,i_t,i_t] + r * r * d1.h_LL[:,i_t,i_t,i_r]
        
        denominator = ep2phi * bar_gamma_LL[:,i_t,i_t] * np.sqrt(bar_gamma_LL[:,i_r,i_r])
        bar_Ktt = A_LL[:,i_t,i_t] + one_third * bar_gamma_LL[:,i_t,i_t] * bssn_vars.K
        
        # End of: Calculate some useful quantities, now start diagnostic
        #################################################################
        
        # Expansion given by eqn. (49) of NRPy+ https://arxiv.org/abs/1712.07658        
        omega[i,:] = ((4.0 * bar_gamma_LL[:,i_t,i_t] * d1.phi[:,i_r] + d1_bargamma_tt_dr) / denominator 
                      - 2.0 * bar_Ktt / bar_gamma_LL[:,i_t,i_t])
        
        # Fix endpoints
        grid.fill_inner_boundary_single_variable(omega[i,:])

        # Find horizon, i.e. zero crossings of omega (r * omega for better convergence)
        omega_i = omega[i,:]
        min_omega = min(omega_i)
        r_min = r[np.where(omega_i == min_omega)][0]
        if min_omega > 0:
            # no horizon
            ah_radius[i] = 0
            bh_mass[i] = 0
        else:
            r_omega_interp = interp1d(r, abs(r) * omega_i)
            gamma_tt_interp = interp1d(r, bar_gamma_LL[:,i_t,i_t] / em4phi)
            gamma_pp_interp = interp1d(r, bar_gamma_LL[:,i_p,i_p] / em4phi)

            # Root finding brentq function
            r_horizon = brentq(r_omega_interp, r_min, r[-1])
                        
            # Area is calculated via the induced metric sqrt(sigma) dtheta dphi
            # Where sigma = diag(gamma_tt, gamma_pp)
            area = 4.0 * np.pi * np.sqrt(gamma_tt_interp(r_horizon) * gamma_pp_interp(r_horizon))
            bh_mass[i] = np.sqrt(area / (16.0 * np.pi))
            ah_radius[i] = r_horizon
    
    # end of iteration over time  
    #########################################################################
    
    return omega, ah_radius, bh_mass