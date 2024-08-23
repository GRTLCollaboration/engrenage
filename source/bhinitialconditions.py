"""
Set the initial conditions for all the variables for an isotropic Schwarzschild BH.

See further details in https://github.com/GRChombo/engrenage/wiki/Running-the-black-hole-example.
"""

import numpy as np

from source.uservariables import *
from source.tensoralgebra import *
from source.grid import *


def get_initial_state(grid: Grid) :
    
    # For readability
    r = grid.r
    N = grid.num_points
                     
    initial_state = np.zeros((NUM_VARS, N))
    (
        u,
        v,
        phi,
        hrr,
        htt,
        hpp,
        K,
        arr,
        att,
        app,
        lambdar,
        shiftr,
        br,
        lapse,
    ) = initial_state

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
    grid.fill_inner_boundary(initial_state)

    # assign lambdar values
    h_tensor = np.array([hrr, htt, hpp])

    dh_dr = grid.get_first_derivative(h_tensor)
        
    # (unscaled) \bar\gamma_ij and \bar\gamma^ij
    bar_gamma_LL = get_metric(r, h_tensor)
    bar_gamma_UU = get_inverse_metric(r, h_tensor)
        
    # The connections Delta^i, Delta^i_jk and Delta_ijk
    Delta_U, Delta_ULL, Delta_LLL  = get_connection(r, bar_gamma_UU, bar_gamma_LL, h_tensor, dh_dr)
    lambdar[:]   = Delta_U[i_r]

    # Fill boundary cells for lambdar
    grid.fill_outer_boundary(initial_state, [idx_lambdar])

    # overwrite inner cells using parity under r -> - r
    grid.fill_inner_boundary(initial_state, [idx_lambdar])
            
    return initial_state.reshape(-1)
