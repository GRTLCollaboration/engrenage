# oscillatoninitialconditions.py

# set the initial conditions for all the variables for an oscillaton
# see further details in https://github.com/GRChombo/engrenage/wiki/Running-the-oscillaton-example

from source.uservariables import *
from source.tensoralgebra import *
from source.fourthorderderivatives import *
from source.logderivatives import *
from source.gridfunctions import *
import numpy as np
from scipy.interpolate import interp1d

def get_initial_state(R, N_r, r_is_logarithmic) :
    
    # Set up grid values
    dx, N, r, logarithmic_dr = setup_grid(R, N_r, r_is_logarithmic)
    
    # predefine some userful quantities
    oneoverlogdr = 1.0 / logarithmic_dr
    oneoverlogdr2 = oneoverlogdr * oneoverlogdr
    oneoverdx  = 1.0 / dx
    oneoverdxsquared = oneoverdx * oneoverdx
    
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
    fill_inner_boundary(initial_state, dx, N, r_is_logarithmic)
    
    if(r_is_logarithmic) :
        dhrrdx = get_logdfdx(hrr, oneoverlogdr)
        dhttdx = get_logdfdx(htt, oneoverlogdr)
        dhppdx = get_logdfdx(hpp, oneoverlogdr)
    else :
        dhrrdx     = get_dfdx(hrr, oneoverdx)
        dhttdx     = get_dfdx(htt, oneoverdx)
        dhppdx     = get_dfdx(hpp, oneoverdx)

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
    fill_outer_boundary_ivar(initial_state, dx, N, r_is_logarithmic, idx_lambdar)

    # overwrite inner cells using parity under r -> - r
    fill_inner_boundary_ivar(initial_state, dx, N, r_is_logarithmic, idx_lambdar)
            
    return r, initial_state
