# oscillatoninitialconditions.py

# set the initial conditions for all the variables for an oscillaton
# see further details in https://github.com/GRChombo/engrenage/wiki/Running-the-oscillaton-example

from source.uservariables import *
from source.tensoralgebra import *
from source.Grid import *
import numpy as np
from scipy.interpolate import interp1d

def get_initial_state(a_grid) :
    
    # For readability
    r = a_grid.r_vector
    N = a_grid.num_points_r
    dx = a_grid.base_dx
    
    initial_state = np.zeros(NUM_VARS * N)
    [u,v,phi,hrr,htt,hpp,K,arr,att,app,lambdar,shiftr,br,lapse] = np.array_split(initial_state, NUM_VARS)
    
    # Get stationary oscillaton data for the vars, in both positive and negative R
    grr0_data    = np.loadtxt("../source/initial_data/grr0.csv")
    lapse0_data  = np.loadtxt("../source/initial_data/lapse0.csv")
    v0_data      = np.loadtxt("../source/initial_data/v0.csv")
    length       = np.size(grr0_data)
    grr0_data    = np.concatenate((np.flip(grr0_data), grr0_data[1:length]))
    lapse0_data  = np.concatenate((np.flip(lapse0_data), lapse0_data[1:length]))
    v0_data      = np.concatenate((np.flip(v0_data), v0_data[1:length]))
    
    # set up grid in radial direction in areal polar coordinates
    dR = 0.01
    assert dR < dx, 'your dx is smaller than the oscillaton data, use fewer points!'
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
