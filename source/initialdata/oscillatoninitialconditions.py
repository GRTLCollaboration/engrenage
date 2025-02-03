"""
Set the initial conditions for all the variables for an oscillaton.

See further details in https://github.com/GRChombo/engrenage/wiki/Running-the-oscillaton-example.
"""

import numpy as np
from scipy.interpolate import interp1d

from bssn.tensoralgebra import *
from core.grid import Grid
from bssn.bssnstatevariables import *
from backgrounds.sphericalbackground import *

def get_initial_state(grid: Grid, background) :

    assert grid.NUM_VARS == 14, "NUM_VARS not correct for bssn + scalar field"    
    
    # For readability
    r = grid.r
    N = grid.num_points
    dr = grid.min_dr
    NUM_VARS = grid.NUM_VARS
    
    initial_state = np.zeros((NUM_VARS, N))
    (
        phi, hrr, htt, hpp,
        K, arr, att, app,
        lambdar, shiftr, br, lapse,         
        u, v
    ) = initial_state
    
    # Get stationary oscillaton data for the vars, in both positive and negative R
    grr0_data    = np.loadtxt("../source/initialdata/oscillaton/grr0.csv")
    lapse0_data  = np.loadtxt("../source/initialdata/oscillaton/lapse0.csv")
    v0_data      = np.loadtxt("../source/initialdata/oscillaton/v0.csv")
    length       = np.size(grr0_data)
    grr0_data    = np.concatenate((np.flip(grr0_data), grr0_data[1:length]))
    lapse0_data  = np.concatenate((np.flip(lapse0_data), lapse0_data[1:length]))
    v0_data      = np.concatenate((np.flip(v0_data), v0_data[1:length]))
    
    # set up grid in radial direction in areal polar coordinates
    dR = 0.01
    assert dR < dr, 'your dr is smaller than the oscillaton data, use fewer points!'
    R = np.linspace(-dR*(length-1), dR*(length-1), num=(length*2-1))
    
    # find interpolating functions for the data
    f_grr   = interp1d(R, grr0_data)
    f_lapse = interp1d(R, lapse0_data)
    f_v     = interp1d(R, v0_data)

    # set the (non zero) scalar field values
    v[:] = f_v(r)
    
    # lapse and spatial metric
    lapse[:] = f_lapse(r)
    grr = f_grr(r)
    gtt_over_r2 = 1.0
    gpp_over_r2sintheta = gtt_over_r2
    phys_gamma_over_r4sin2theta = grr * gtt_over_r2 * gpp_over_r2sintheta

    # Work out the rescaled quantities
    # Note sign error in Baumgarte eqn (2), conformal factor
    phi[:] = 1.0/12.0 * np.log(phys_gamma_over_r4sin2theta)
    em4phi = np.exp(-4.0*phi)
    hrr[:] = em4phi * grr - 1.0
    htt[:] = em4phi * gtt_over_r2 - 1.0
    hpp[:] = em4phi * gpp_over_r2sintheta - 1.0
    
    # overwrite inner cells using parity under r -> - r
    grid.fill_inner_boundary(initial_state)

    # Set up matrices
    zeros = np.zeros_like(hrr)
    h_LL = np.array([[hrr, zeros, zeros],[zeros, htt, zeros],[zeros, zeros, hpp]])
    h_LL = np.moveaxis(h_LL, -1, 0) 
    first_derivative_indices = [idx_hrr, idx_htt, idx_hpp]
    dstate_dr = grid.get_first_derivative(initial_state, first_derivative_indices)
    (dhrr_dr, dhtt_dr, dhpp_dr) = dstate_dr[first_derivative_indices]

    # This is d h_ij / dx^k = dh_dx[x,i,j,k]
    d1_h_dx = np.zeros([N, SPACEDIM, SPACEDIM, SPACEDIM])
    d1_h_dx[:,i_r,i_r, i_r]  = dhrr_dr
    d1_h_dx[:,i_t,i_t, i_r]  = dhtt_dr
    d1_h_dx[:,i_p,i_p, i_r]  = dhpp_dr
        
    # (unscaled) \bar\gamma_ij and \bar\gamma^ij
    bar_gamma_LL = get_bar_gamma_LL(r, h_LL, background)
    bar_gamma_UU = get_bar_gamma_UU(r, h_LL, background)
        
    # The connections Delta^i, Delta^i_jk and Delta_ijk
    Delta_U, Delta_ULL, Delta_LLL  = get_tensor_connections(r, h_LL, d1_h_dx, background)
    lambdar[:]   = Delta_U[:,i_r]

    # Fill boundary cells for lambdar
    grid.fill_outer_boundary(initial_state, [idx_lambdar])

    # overwrite inner cells using parity under r -> - r
    grid.fill_inner_boundary(initial_state, [idx_lambdar])
            
    return initial_state.reshape(-1)
