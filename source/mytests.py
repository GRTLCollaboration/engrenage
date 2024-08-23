# mytests.py

# File providing test data for the tests - these are solutions where the curvature quantities are known
# so provide a test that everything is working ok

import  numpy as np

from source.grid import Grid
from source.uservariables import *
from source.tensoralgebra import *


# This routine gives us something where phi = 0 initially but bar_R and lambda are non trivial
def get_test_state_1(grid: Grid):
    
    # For readability
    r = grid.r
    N = grid.num_points

    test_state = np.zeros((NUM_VARS, N))
    (
        u, v,
        phi, hrr, htt, hpp,
        K, arr, att, app,
        lambdar, shiftr, br, lapse,
    ) = test_state
    
    # lapse and spatial metric
    lapse.fill(1.0)
    grr = 1.0 + r * r * np.exp(-r)
    gtt_over_r2 = grr**(-0.5)
    gpp_over_r2sintheta = gtt_over_r2
    phys_gamma_over_r4sin2theta = grr * gtt_over_r2 * gpp_over_r2sintheta

    # Work out the rescaled quantities
    # Note sign error in Baumgarte eqn (2), conformal factor
    phi[:] = 1.0/12.0 * np.log(phys_gamma_over_r4sin2theta)
    em4phi = np.exp(-4.0*phi)
    hrr[:] = em4phi * grr - 1.0
    htt[:] = em4phi * gtt_over_r2 - 1.0
    hpp[:] = em4phi * gpp_over_r2sintheta - 1.0   
        
    # overwrite outer boundaries with extrapolation (zeroth order)
    for ivar in range(0, NUM_VARS) :
        boundary_cells = np.array([(ivar + 1)*N-3, (ivar + 1)*N-2, (ivar + 1)*N-1])
        for count, ix in enumerate(boundary_cells) :
            offset = -1 - count
            test_state[ix]    = test_state[ix + offset]

    # overwrite inner cells using parity under r -> - r
    grid.fill_inner_boundary(test_state)

    h_tensor = np.array([hrr, htt, hpp])
    dh_dr = grid.get_first_derivative(h_tensor)
        
    # (unscaled) \bar\gamma_ij and \bar\gamma^ij
    bar_gamma_LL = get_metric(r, h_tensor)
    bar_gamma_UU = get_inverse_metric(r, h_tensor)
        
    # The connections Delta^i, Delta^i_jk and Delta_ijk
    Delta_U, Delta_ULL, Delta_LLL  = get_connection(r, bar_gamma_UU, bar_gamma_LL, h_tensor, dh_dr)
    lambdar[:]   = Delta_U[i_r]

    # Fill boundary cells for lambdar
    grid.fill_outer_boundary(test_state, [idx_lambdar])

    # overwrite inner cells using parity under r -> - r
    grid.fill_inner_boundary(test_state, [idx_lambdar])
            
    return test_state.reshape(-1)


# This routine gives us something where bar_R is trivial but phi is non trivial
def get_test_state_2(grid: Grid):
    
    # For readability
    r = grid.r
    N = grid.num_points

    test_state = np.zeros((NUM_VARS, N))
    (
        u, v,
        phi, hrr, htt, hpp,
        K, arr, att, app,
        lambdar, shiftr, br, lapse,
    ) = test_state

    # lapse and spatial metric
    lapse.fill(1.0)
    grr = 1.0 + r * r * np.exp(-r)
    gtt_over_r2 = grr
    gpp_over_r2sintheta = gtt_over_r2
    phys_gamma_over_r4sin2theta = grr * gtt_over_r2 * gpp_over_r2sintheta

    # Work out the rescaled quantities
    # Note sign error in Baumgarte eqn (2), conformal factor
    phi[:] = 1.0/12.0 * np.log(phys_gamma_over_r4sin2theta)
    em4phi = np.exp(-4.0*phi)
    hrr[:] = em4phi * grr - 1.0
    htt[:] = em4phi * gtt_over_r2 - 1.0
    hpp[:] = em4phi * gpp_over_r2sintheta - 1.0 
        
    # overwrite outer boundaries with extrapolation (zeroth order)
    for ivar in range(0, NUM_VARS) :
        boundary_cells = np.array([(ivar + 1)*N-3, (ivar + 1)*N-2, (ivar + 1)*N-1])
        for count, ix in enumerate(boundary_cells) :
            offset = -1 - count
            test_state[ix]    = test_state[ix + offset]

    # overwrite inner cells using parity under r -> - r
    grid.fill_inner_boundary(test_state)

    h_tensor = np.array([hrr, htt, hpp])
    dh_dr = grid.get_first_derivative(h_tensor)
        
    # (unscaled) \bar\gamma_ij and \bar\gamma^ij
    bar_gamma_LL = get_metric(r, h_tensor)
    bar_gamma_UU = get_inverse_metric(r, h_tensor)
        
    # The connections Delta^i, Delta^i_jk and Delta_ijk
    Delta_U, Delta_ULL, Delta_LLL  = get_connection(r, bar_gamma_UU, bar_gamma_LL, h_tensor, dh_dr)
    lambdar[:]   = Delta_U[i_r]

    # Fill boundary cells for lambdar
    grid.fill_outer_boundary(test_state, [idx_lambdar])

    # overwrite inner cells using parity under r -> - r
    grid.fill_inner_boundary(test_state, [idx_lambdar])
            
    return test_state.reshape(-1)


# This routine gives us the Schwazschild metric in the original ingoing Eddington Finkelstien coords
# that is r = r_schwarzschild and t = t_schwarzschild - (r-r*)
# For this the RHS should be zero, but unlike in Schwarschild coords Kij and the shift are non trivial
# (thanks to Ulrich Sperhake for suggesting this test)
def get_test_state_bh(grid: Grid):
    
    # For readability
    r = grid.r
    N = grid.num_points
    
    test_state = np.zeros((NUM_VARS, N))
    (
        u, v,
        phi, hrr, htt, hpp,
        K, arr, att, app,
        lambdar, shiftr, br, lapse,
    ) = test_state
    GM = 1.0
    
    # lapse, shift and spatial metric
    H = 2.0 * GM / abs(r)
    dHdr = - 2.0 * GM / r / r
    lapse[:] = 1.0/np.sqrt(1.0 + H)
    grr = 1.0 + H
    shiftr[:] = H / grr
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

    # These are \Gamma^i_jk
    chris_rrr = 0.5 * dHdr / grr
    chris_rtt = - r / grr
    chris_rpp = chris_rtt * sin2theta

    #K_ij = (D_i shift_j + D_j shift_i) / lapse / 2 (since dgammadt = 0)        
    Krr = (dHdr - chris_rrr * H) / lapse
    Ktt_over_r2 = - chris_rtt * H / lapse / r / r # 2.0 * lapse / r_i / r_i
    Kpp_over_r2sintheta = - chris_rpp * H / lapse / r / r / sin2theta
    K[:] = Krr / grr + Ktt_over_r2 / gtt_over_r2  + Kpp_over_r2sintheta / gpp_over_r2sintheta
    arr[:] = em4phi * (Krr - 1.0/3.0 * grr * K)
    att[:] = em4phi * (Ktt_over_r2 - 1.0/3.0 * gtt_over_r2 * K)
    app[:] = em4phi * (Kpp_over_r2sintheta - 1.0/3.0 * gpp_over_r2sintheta * K)
       
    # overwrite inner cells using parity under r -> - r
    grid.fill_inner_boundary(test_state)
    
    h_tensor = np.array([hrr, htt, hpp])
    dh_dr = grid.get_first_derivative(h_tensor)
        
    # (unscaled) \bar\gamma_ij and \bar\gamma^ij
    bar_gamma_LL = get_metric(r, h_tensor)
    bar_gamma_UU = get_inverse_metric(r, h_tensor)
        
    # The connections Delta^i, Delta^i_jk and Delta_ijk
    Delta_U, Delta_ULL, Delta_LLL  = get_connection(r, bar_gamma_UU, bar_gamma_LL, h_tensor, dh_dr)
    lambdar[:]   = Delta_U[i_r]

    # Fill boundary cells for lambdar
    grid.fill_outer_boundary(test_state, [idx_lambdar])

    # overwrite inner cells using parity under r -> - r
    grid.fill_inner_boundary(test_state, [idx_lambdar])
            
    return test_state.reshape(-1)
