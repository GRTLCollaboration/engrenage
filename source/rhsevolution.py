#rhsevolution.py

# python modules
import numpy as np
import time

# homemade code
from source.uservariables import *
from source.tensoralgebra import *
from source.mymatter import *
from source.bssnrhs import *
from source.grid import Grid


# function that returns the rhs for each of the field vars
# see further details in https://github.com/GRChombo/engrenage/wiki/Useful-code-background
def get_rhs(t_i, current_state: np.ndarray, grid: Grid, eta, progress_bar, time_state) :

    # Uncomment for timing and tracking progress
    # start_time = time.time()
    
    # For readability
    r = grid.r
    
    ####################################################################################################
    #unpackage the state vector for readability - these are the vectors of values across r values at time t_i
    # see uservariables.py for naming conventions
    
    # Unpack variables from current_state - see uservariables.py
    u, v , phi, hrr, htt, hpp, K, arr, att, app, lambdar, shiftr, br, lapse = np.array_split(current_state, NUM_VARS)
    state = current_state.reshape(NUM_VARS, -1)

    # this is where the rhs will go
    rhs = np.zeros_like(state)

    ####################################################################################################
    # enforce that the determinant of \bar gamma_ij is equal to that of flat space in spherical coords
    # (note that trace of \bar A_ij = 0 is enforced dynamically below as in Etienne https://arxiv.org/abs/1712.07658v2)
    
    h = np.array([hrr, htt, hpp])
    determinant = abs(get_rescaled_determinant_gamma(h))
        
    hrr = (1.0 + hrr)/ np.power(determinant,1./3) - 1.0
    htt = (1.0 + htt)/ np.power(determinant,1./3) - 1.0
    hpp = (1.0 + hpp)/ np.power(determinant,1./3) - 1.0
    ####################################################################################################
    
    # second derivatives
    second_derivative_indices = [idx_u, idx_phi, idx_hrr, idx_htt, idx_hpp, idx_shiftr, idx_lapse]
    d2state_dr2 = grid.get_second_derivative(state, second_derivative_indices)

    (
        d2u_dr2,
        d2phi_dr2,
        d2hrr_dr2,
        d2htt_dr2,
        d2hpp_dr2,
        d2shiftr_dr2,
        d2lapse_dr2,
    ) = d2state_dr2[second_derivative_indices]

    
    # First derivatives
    first_derivative_indices = [idx_u, idx_phi, idx_hrr, idx_htt, idx_hpp, idx_K, idx_lambdar, idx_shiftr, idx_lapse]
    dstate_dr = grid.get_first_derivative(state, first_derivative_indices)

    (
        du_dr,
        dphi_dr,
        dhrr_dr,
        dhtt_dr,
        dhpp_dr,
        dK_dr,
        dlambdar_dr,
        dshiftr_dr,
        dlapse_dr,
    ) = dstate_dr[first_derivative_indices]
    
    # First derivatives - advec left and right along shiftr (left if shiftr < 0 and right if shiftr >= 0)
    advec_indices = [idx_u, idx_v, idx_phi, idx_hrr, idx_htt, idx_hpp, idx_arr, idx_att, idx_app, idx_K, idx_lambdar]
    dstate_dr_advec = grid.get_advection(state, shiftr >= 0, advec_indices)

    (
        du_dr_advec,
        dv_dr_advec,
        dphi_dr_advec,
        dhrr_dr_advec,
        dhtt_dr_advec,
        dhpp_dr_advec,
        darr_dr_advec,
        datt_dr_advec,
        dapp_dr_advec,
        dK_dr_advec,
        dlambdar_dr_advec,
    ) = dstate_dr_advec[advec_indices]
        
    ####################################################################################################
    
    # assign parts of the rhs vector to the different vars
    (
        rhs_u,
        rhs_v,
        rhs_phi,
        rhs_hrr,
        rhs_htt,
        rhs_hpp,
        rhs_K,
        rhs_arr,
        rhs_att,
        rhs_app,
        rhs_lambdar,
        rhs_shiftr,
        rhs_br,
        rhs_lapse,
    ) = rhs

    ####################################################################################################     
    # now calculate the rhs values for the main grid (boundaries handled below)       
    a = np.array([arr, att, app])
    em4phi = np.exp(-4.0*phi)
    dhdr   = np.array([dhrr_dr, dhtt_dr, dhpp_dr])
    d2hdr2 = np.array([d2hrr_dr2, d2htt_dr2, d2hpp_dr2])
       
    # Calculate some useful quantities
    # (mostly from tensoralgebra.py)
    ########################################################
        
    # rescaled \bar\gamma_ij and \bar\gamma^ij
    r_gamma_LL = get_rescaled_metric(h)
    r_gamma_UU = get_rescaled_inverse_metric(h)
        
    # \bar A_ij, \bar A^ij and the trace A_i^i, then Asquared = \bar A_ij \bar A^ij
    a_UU = get_a_UU(a, r_gamma_UU)
    traceA   = get_trace_A(a, r_gamma_UU)
    Asquared = get_Asquared(a, r_gamma_UU)
        
    # The rescaled connections Delta^i, Delta^i_jk and Delta_ijk
    rDelta_U, rDelta_ULL, rDelta_LLL  = get_rescaled_connection(r, r_gamma_UU, 
                                                                r_gamma_LL, h, dhdr)
    # rescaled \bar \Gamma^i_jk
    r_conformal_chris = get_rescaled_conformal_chris(rDelta_ULL, r)
        
    # rescaled Ricci tensor
    rbar_Rij = get_rescaled_ricci_tensor(r, h, dhdr, d2hdr2, lambdar, dlambdar_dr,
                                         rDelta_U, rDelta_ULL, rDelta_LLL, 
                                         r_gamma_UU, r_gamma_LL)
        
    # This is the conformal divergence of the shift \bar D_i \beta^i
    # Use the fact that the conformal metric determinant is \hat \gamma = r^4 sin2theta
    bar_div_shift =  (dshiftr_dr + 2.0 * shiftr / r)
                
    # Matter sources - see mymatter.py
    matter_rho             = get_rho( u, du_dr, v, r_gamma_UU, em4phi )
    matter_Si              = get_Si(  u, du_dr, v)
    matter_S, matter_rSij  = get_rescaled_Sij( u, du_dr, v, r_gamma_UU, em4phi, r_gamma_LL)

        
    # End of: Calculate some useful quantities, now start RHS
    #########################################################

    # Get the matter rhs - see mymatter.py
    rhs_u[:], rhs_v[:] = get_matter_rhs(u, v, du_dr, d2u_dr2,
                                        r_gamma_UU, em4phi, dphi_dr,
                                        K, lapse, dlapse_dr, r_conformal_chris)

    # Get the bssn rhs - see bssnrhs.py
    rhs_phi[:]  = get_rhs_phi(lapse, K, bar_div_shift)
        
    rhs_h       = get_rhs_h(r, r_gamma_LL, lapse, traceA, dshiftr_dr, shiftr, bar_div_shift, a)
        
    rhs_K[:]    = get_rhs_K(lapse, K, Asquared, em4phi, d2lapse_dr2, dlapse_dr,
                               r_conformal_chris, dphi_dr, r_gamma_UU, matter_rho, matter_S)
        
    rhs_a       = get_rhs_a(r, a, bar_div_shift, lapse, K, em4phi, rbar_Rij,
                               r_conformal_chris, r_gamma_UU, r_gamma_LL,
                               d2phi_dr2, dphi_dr, d2lapse_dr2, dlapse_dr,
                               h, dhdr, d2hdr2, matter_rSij)
        
    rhs_lambdar[:] = get_rhs_lambdar(r, d2shiftr_dr2, dshiftr_dr, shiftr, h, dhdr,
                                     rDelta_U, rDelta_ULL, bar_div_shift,
                                     r_gamma_UU, a_UU, lapse,
                                     dlapse_dr, dphi_dr, dK_dr, matter_Si)
        
    # Set the gauge vars rhs
    # eta is the 1+log slicing damping coefficient - of order 1/M_adm of spacetime        
    rhs_br[:]     = 0.75 * rhs_lambdar - eta * br
    rhs_shiftr[:] = br
    rhs_lapse[:]  = - 2.0 * lapse * K        
        
    # Add advection to time derivatives (this is the bit coming from the Lie derivative)
    # Note the additional advection terms from rescaling for tensors

    rhs_u[:]       += shiftr * du_dr_advec
    rhs_v[:]       += shiftr * dv_dr_advec
    rhs_phi[:]     += shiftr * dphi_dr_advec
    rhs_hrr[:]     = rhs_h[i_r][i_r] + shiftr * dhrr_dr_advec + 2.0 * hrr * dshiftr_dr
    rhs_htt[:]     = rhs_h[i_t][i_t] + shiftr * dhtt_dr_advec + 2.0 * shiftr * 1.0/r * htt
    rhs_hpp[:]     = rhs_h[i_p][i_p] + shiftr * dhpp_dr_advec + 2.0 * shiftr * 1.0/r * hpp
    rhs_K[:]       += shiftr * dK_dr_advec
    rhs_arr[:]     = rhs_a[i_r][i_r] + shiftr * darr_dr_advec + 2.0 * arr * dshiftr_dr
    rhs_att[:]     = rhs_a[i_t][i_t] + shiftr * datt_dr_advec + 2.0 * shiftr * 1.0/r * att
    rhs_app[:]     = rhs_a[i_p][i_p] + shiftr * dapp_dr_advec + 2.0 * shiftr * 1.0/r * app
    rhs_lambdar[:] += shiftr * dlambdar_dr_advec - lambdar * dshiftr_dr

    # end of rhs iteration   
    
    ####################################################################################################
            
    # finally add Kreiss Oliger dissipation which removes noise at frequency of grid resolution
    sigma = 0 # kreiss-oliger damping coefficient, max_step should be limited to avoid instability
    
    diss = sigma * grid.get_kreiss_oliger_diss(state)
    rhs += sigma * diss

    #################################################################################################### 
    
    # see https://github.com/KAClough/BabyGRChombo/wiki/Useful-code-background
    
    # overwrite outer boundaries with extrapolation (order specified in uservariables.py)
    grid.fill_outer_boundary(rhs)

    # overwrite inner cells using parity under r -> - r
    grid.fill_inner_boundary(rhs)
    #################################################################################################### 
    
    # Some code for checking timing and progress output
    
    # state is a list containing last updated time t:
    # state = [last_t, dt for progress bar]
    # its values can be carried between function calls throughout the ODE integration
    last_t, deltat = time_state
    
    # call update(n) here where n = (t - last_t) / dt
    n = int((t_i - last_t)/deltat)
    progress_bar.update(n)
    # we need this to take into account that n is a rounded number:
    time_state[0] = last_t + deltat * n 
    
    # end_time = time.time()
    # print("total rhs time at t= ", t_i, " is, ", end_time-start_time)
        
    ####################################################################################################
    
    # Finally return the rhs
    return rhs.reshape(-1)
