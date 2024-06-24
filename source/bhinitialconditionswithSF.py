# bhinitialconditions.py

# set the initial conditions for all the variables for an isotropic Schwarzschild BH
# see further details in https://github.com/GRChombo/engrenage/wiki/Running-the-black-hole-example

from source.uservariables import *
from source.tensoralgebra import *
from source.Grid import *
import numpy as np
from scipy.interpolate import interp1d

def get_initial_state(a_grid) :
    
    # For readability
    r = a_grid.r_vector
    N = a_grid.num_points_r
                     
    initial_state = np.zeros(NUM_VARS * N)
    [u,v,phi,hrr,htt,hpp,K,arr,att,app,lambdar,shiftr,br,lapse] = np.array_split(initial_state, NUM_VARS)

    # Set BH length scale
    GM = 1.0
    
    # set non zero metric values
    grr = (1 + 0.5 * GM/r)**4.0
    gtt_over_r2 = grr
    gpp_over_r2sintheta = gtt_over_r2
    phys_gamma_over_r4sin2theta = grr * gtt_over_r2 * gpp_over_r2sintheta
    # Note sign error in Baumgarte eqn (2), conformal factor
    phi[:] = 1.0/12.0 * np.log(phys_gamma_over_r4sin2theta)
    # Cap the phi value in the centre to stop unphysically large numbers at singularity
    phi[:] = np.clip(phi, None, 10.0)
    em4phi = np.exp(-4.0*phi)
    hrr[:] = em4phi * grr - 1.0
    htt[:] = em4phi * gtt_over_r2 - 1.0
    hpp[:] = em4phi * gpp_over_r2sintheta - 1.0    
    lapse.fill(1.0)
    #lapse[:] = em4phi # optional, to pre collapse the lapse
    
    # overwrite inner cells using parity under r -> - r
    a_grid.fill_inner_boundary(initial_state)
             
    dhrrdx     = np.dot(a_grid.derivatives.d1_matrix, hrr)
    dhttdx     = np.dot(a_grid.derivatives.d1_matrix, htt)
    dhppdx     = np.dot(a_grid.derivatives.d1_matrix, hpp)

    # assign lambdar values
    h_tensor = np.array([hrr, htt, hpp])
    a_tensor = np.array([arr, att, app])
    dhdr   = np.array([dhrrdx, dhttdx, dhppdx])
        
    # (unscaled) \bar\gamma_ij and \bar\gamma^ij
    bar_gamma_LL = get_metric(r, h_tensor)
    bar_gamma_UU = get_inverse_metric(r, h_tensor)
        
    # The connections Delta^i, Delta^i_jk and Delta_ijk
    Delta_U, Delta_ULL, Delta_LLL  = get_connection(r, bar_gamma_UU, bar_gamma_LL, h_tensor, dhdr)
    lambdar[:]   = Delta_U[i_r]

    # Fill boundary cells for lambdar
    a_grid.fill_outer_boundary_ivar(initial_state, idx_lambdar)

    # overwrite inner cells using parity under r -> - r
    a_grid.fill_inner_boundary_ivar(initial_state, idx_lambdar)
            
    return initial_state
