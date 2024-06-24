#rhsevolution.py

# python modules
import numpy as np
import time

# homemade code
from source.uservariables import *
from source.tensoralgebra import *
from source.mymatter import *
from source.bssnrhs import *
from source.Grid import *

# function that returns the rhs for each of the field vars
# see further details in https://github.com/GRChombo/engrenage/wiki/Useful-code-background
def get_rhs(t_i, current_state, my_grid, eta, progress_bar, time_state) :

    # Uncomment for timing and tracking progress
    # start_time = time.time()
    
    # For readability
    r = my_grid.r_vector
    N = my_grid.num_points_r
    
    # this is where the rhs will go
    rhs = np.zeros_like(current_state) 
    
 ####################################################################################################
    #unpackage the state vector for readability - these are the vectors of values across r values at time t_i
    # see uservariables.py for naming conventions
    
    # Unpack variables from current_state - see uservariables.py
    u, v , phi, hrr, htt, hpp, K, arr, att, app, lambdar, shiftr, br, lapse = np.array_split(current_state, NUM_VARS) 
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
    d2udx2      = np.dot(my_grid.derivatives.d2_matrix, u     )
    d2phidx2    = np.dot(my_grid.derivatives.d2_matrix, phi   )
    d2hrrdx2    = np.dot(my_grid.derivatives.d2_matrix, hrr   )
    d2httdx2    = np.dot(my_grid.derivatives.d2_matrix, htt   )
    d2hppdx2    = np.dot(my_grid.derivatives.d2_matrix, hpp   )
    d2lapsedx2  = np.dot(my_grid.derivatives.d2_matrix, lapse )
    d2shiftrdx2 = np.dot(my_grid.derivatives.d2_matrix, shiftr)
    
    # first derivatives
    dudx       = np.dot(my_grid.derivatives.d1_matrix, u      )
    dvdx       = np.dot(my_grid.derivatives.d1_matrix, v      )
    dphidx     = np.dot(my_grid.derivatives.d1_matrix, phi    )
    dhrrdx     = np.dot(my_grid.derivatives.d1_matrix, hrr    )
    dhttdx     = np.dot(my_grid.derivatives.d1_matrix, htt    )
    dhppdx     = np.dot(my_grid.derivatives.d1_matrix, hpp    )
    darrdx     = np.dot(my_grid.derivatives.d1_matrix, arr    )
    dattdx     = np.dot(my_grid.derivatives.d1_matrix, att    )
    dappdx     = np.dot(my_grid.derivatives.d1_matrix, app    )
    dKdx       = np.dot(my_grid.derivatives.d1_matrix, K      )
    dlambdardx = np.dot(my_grid.derivatives.d1_matrix, lambdar)
    dshiftrdx  = np.dot(my_grid.derivatives.d1_matrix, shiftr )
    dbrdx      = np.dot(my_grid.derivatives.d1_matrix, br     )
    dlapsedx   = np.dot(my_grid.derivatives.d1_matrix, lapse  )
    
    # first derivatives - advec left and right
    dudx_advec_L       = np.dot(my_grid.derivatives.d_advec_matrix_left, u      )
    dvdx_advec_L       = np.dot(my_grid.derivatives.d_advec_matrix_left, v      )
    dphidx_advec_L     = np.dot(my_grid.derivatives.d_advec_matrix_left, phi    )
    dhrrdx_advec_L     = np.dot(my_grid.derivatives.d_advec_matrix_left, hrr    )
    dhttdx_advec_L     = np.dot(my_grid.derivatives.d_advec_matrix_left, htt    )
    dhppdx_advec_L     = np.dot(my_grid.derivatives.d_advec_matrix_left, hpp    )
    darrdx_advec_L     = np.dot(my_grid.derivatives.d_advec_matrix_left, arr    )
    dattdx_advec_L     = np.dot(my_grid.derivatives.d_advec_matrix_left, att    )
    dappdx_advec_L     = np.dot(my_grid.derivatives.d_advec_matrix_left, app    )
    dKdx_advec_L       = np.dot(my_grid.derivatives.d_advec_matrix_left, K      )
    dlambdardx_advec_L = np.dot(my_grid.derivatives.d_advec_matrix_left, lambdar)
    dshiftrdx_advec_L  = np.dot(my_grid.derivatives.d_advec_matrix_left, shiftr )
    dbrdx_advec_L      = np.dot(my_grid.derivatives.d_advec_matrix_left, br     )
    dlapsedx_advec_L   = np.dot(my_grid.derivatives.d_advec_matrix_left, lapse  )
    
    dudx_advec_R       = np.dot(my_grid.derivatives.d_advec_matrix_right, u      )
    dvdx_advec_R       = np.dot(my_grid.derivatives.d_advec_matrix_right, v      )
    dphidx_advec_R     = np.dot(my_grid.derivatives.d_advec_matrix_right, phi    )
    dhrrdx_advec_R     = np.dot(my_grid.derivatives.d_advec_matrix_right, hrr    )
    dhttdx_advec_R     = np.dot(my_grid.derivatives.d_advec_matrix_right, htt    )
    dhppdx_advec_R     = np.dot(my_grid.derivatives.d_advec_matrix_right, hpp    )
    darrdx_advec_R     = np.dot(my_grid.derivatives.d_advec_matrix_right, arr    )
    dattdx_advec_R     = np.dot(my_grid.derivatives.d_advec_matrix_right, att    )
    dappdx_advec_R     = np.dot(my_grid.derivatives.d_advec_matrix_right, app    )
    dKdx_advec_R       = np.dot(my_grid.derivatives.d_advec_matrix_right, K      )
    dlambdardx_advec_R = np.dot(my_grid.derivatives.d_advec_matrix_right, lambdar)
    dshiftrdx_advec_R  = np.dot(my_grid.derivatives.d_advec_matrix_right, shiftr )
    dbrdx_advec_R      = np.dot(my_grid.derivatives.d_advec_matrix_right, br     )
    dlapsedx_advec_R   = np.dot(my_grid.derivatives.d_advec_matrix_right, lapse  ) 
        
    ####################################################################################################
    
    # assign parts of the rhs vector to the different vars
    rhs_u, rhs_v, rhs_phi, rhs_hrr, rhs_htt, rhs_hpp, rhs_K, rhs_arr, rhs_att, rhs_app, rhs_lambdar, rhs_shiftr, rhs_br, rhs_lapse = np.array_split(rhs, NUM_VARS)    
    ####################################################################################################     
    # now calculate the rhs values for the main grid (boundaries handled below)       
    a = np.array([arr, att, app])
    em4phi = np.exp(-4.0*phi)
    dhdr   = np.array([dhrrdx, dhttdx, dhppdx])
    d2hdr2 = np.array([d2hrrdx2, d2httdx2, d2hppdx2])
       
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
    rbar_Rij = get_rescaled_ricci_tensor(r, h, dhdr, d2hdr2, lambdar, dlambdardx,
                                         rDelta_U, rDelta_ULL, rDelta_LLL, 
                                         r_gamma_UU, r_gamma_LL)
        
    # This is the conformal divergence of the shift \bar D_i \beta^i
    # Use the fact that the conformal metric determinant is \hat \gamma = r^4 sin2theta
    bar_div_shift =  (dshiftrdx + 2.0 * shiftr / r)
                
    # Matter sources - see mymatter.py
    matter_rho             = get_rho( u, dudx, v, r_gamma_UU, em4phi )
    matter_Si              = get_Si(  u, dudx, v)
    matter_S, matter_rSij  = get_rescaled_Sij( u, dudx, v, r_gamma_UU, em4phi, r_gamma_LL)

        
    # End of: Calculate some useful quantities, now start RHS
    #########################################################

    # Get the matter rhs - see mymatter.py
    rhs_u[:], rhs_v[:] = get_matter_rhs(u, v, dudx, d2udx2, 
                                        r_gamma_UU, em4phi, dphidx, 
                                        K, lapse, dlapsedx, r_conformal_chris)

    # Get the bssn rhs - see bssnrhs.py
    rhs_phi[:]  = get_rhs_phi(lapse, K, bar_div_shift)
        
    rhs_h       = get_rhs_h(r, r_gamma_LL, lapse, traceA, dshiftrdx, shiftr, bar_div_shift, a)
        
    rhs_K[:]    = get_rhs_K(lapse, K, Asquared, em4phi, d2lapsedx2, dlapsedx, 
                               r_conformal_chris, dphidx, r_gamma_UU, matter_rho, matter_S)
        
    rhs_a       = get_rhs_a(r, a, bar_div_shift, lapse, K, em4phi, rbar_Rij,
                               r_conformal_chris, r_gamma_UU, r_gamma_LL,
                               d2phidx2, dphidx, d2lapsedx2, dlapsedx, 
                               h, dhdr, d2hdr2, matter_rSij)
        
    rhs_lambdar[:] = get_rhs_lambdar(r, d2shiftrdx2, dshiftrdx, shiftr, h, dhdr,
                                     rDelta_U, rDelta_ULL, bar_div_shift,
                                     r_gamma_UU, a_UU, lapse,
                                     dlapsedx, dphidx, dKdx, matter_Si)
        
    # Set the gauge vars rhs
    # eta is the 1+log slicing damping coefficient - of order 1/M_adm of spacetime        
    rhs_br[:]     = 0.75 * rhs_lambdar - eta * br
    rhs_shiftr[:] = br

    # 1 + log slicing old lapse condition
    #rhs_lapse[:]  = - 2.0 * lapse * K
    
    # Implement the new shock avoiding lapse condition from
    # https://inspirehep.net/literature/2111279
    kappa_lapse = 0.05
    rhs_lapse[:]  = - (lapse * lapse + kappa_lapse) * K
        
    # Add advection to time derivatives (this is the bit coming from the Lie derivative)
    # Note the additional advection terms from rescaling for tensors
    for ix in range(N) :
        
        r_here = r[ix]
        
        if (shiftr[ix] > 0) :
            rhs_u[ix]       += shiftr[ix] * dudx_advec_R[ix]
            rhs_v[ix]       += shiftr[ix] * dvdx_advec_R[ix]
            rhs_phi[ix]     += shiftr[ix] * dphidx_advec_R[ix]
            rhs_hrr[ix]     = (rhs_h[i_r][i_r][ix] + shiftr[ix] * dhrrdx_advec_R[ix] 
                               + 2.0 * hrr[ix] * dshiftrdx[ix])
            rhs_htt[ix]     = (rhs_h[i_t][i_t][ix] + shiftr[ix] * dhttdx_advec_R[ix]
                               + 2.0 * shiftr[ix] * 1.0/r_here * htt[ix])
            rhs_hpp[ix]     = (rhs_h[i_p][i_p][ix] + shiftr[ix] * dhppdx_advec_R[ix]
                               + 2.0 * shiftr[ix] * 1.0/r_here * hpp[ix])
            rhs_K[ix]       += shiftr[ix] * dKdx_advec_R[ix]
            rhs_arr[ix]     = (rhs_a[i_r][i_r][ix] + shiftr[ix] * darrdx_advec_R[ix] 
                               + 2.0 * arr[ix] * dshiftrdx[ix])
            rhs_att[ix]     = (rhs_a[i_t][i_t][ix] + shiftr[ix] * dattdx_advec_R[ix]
                               + 2.0 * shiftr[ix] * 1.0/r_here * att[ix])
            rhs_app[ix]     = (rhs_a[i_p][i_p][ix] + shiftr[ix] * dappdx_advec_R[ix]
                               + 2.0 * shiftr[ix] * 1.0/r_here * app[ix])
            rhs_lambdar[ix] += (shiftr[ix] * dlambdardx_advec_R[ix] 
                                - lambdar[ix] * dshiftrdx[ix])
            # NB optional to add advection to lapse and shift vars
            # rhs_lapse       += shiftr[ix] * dlapsedx_advec_R[ix]
            # rhs_br[ix]      += 0.0
            # rhs_shiftr[ix]  += 0.0
            
        else : 
            rhs_u[ix]       += shiftr[ix] * dudx_advec_L[ix]
            rhs_v[ix]       += shiftr[ix] * dvdx_advec_L[ix]
            rhs_phi[ix]     += shiftr[ix] * dphidx_advec_L[ix]
            rhs_hrr[ix]     = (rhs_h[i_r][i_r][ix] + shiftr[ix] * dhrrdx_advec_L[ix]
                               + 2.0 * hrr[ix] * dshiftrdx[ix])
            rhs_htt[ix]     = (rhs_h[i_t][i_t][ix] + shiftr[ix] * dhttdx_advec_L[ix]
                               + 2.0 * shiftr[ix] * 1.0/r_here * htt[ix])
            rhs_hpp[ix]     = (rhs_h[i_p][i_p][ix] + shiftr[ix] * dhppdx_advec_L[ix]
                               + 2.0 * shiftr[ix] * 1.0/r_here * hpp[ix])
            rhs_K[ix]       += shiftr[ix] * dKdx_advec_L[ix]
            rhs_arr[ix]     = (rhs_a[i_r][i_r][ix] + shiftr[ix] * darrdx_advec_L[ix] 
                               + 2.0 * arr[ix] * dshiftrdx[ix])
            rhs_att[ix]     = (rhs_a[i_t][i_t][ix] + shiftr[ix] * dattdx_advec_L[ix]
                               + 2.0 * shiftr[ix] * 1.0/r_here * att[ix])
            rhs_app[ix]     = (rhs_a[i_p][i_p][ix] + shiftr[ix] * dappdx_advec_L[ix]
                               + 2.0 * shiftr[ix] * 1.0/r_here * app[ix])
            rhs_lambdar[ix] += (shiftr[ix] * dlambdardx_advec_L[ix] 
                                - lambdar[ix] * dshiftrdx[ix])
            # NB optional to add advection to lapse and shift vars
            # rhs_lapse       += shiftr[ix] * dlapsedx_advec_L[ix]            
            # rhs_br[ix]      += 0.0
            # rhs_shiftr[ix]  += 0.0

            
    # end of rhs iteration   
    
 #################################################################################################### 
            
    # finally add Kreiss Oliger dissipation which removes noise at frequency of grid resolution
    sigma = 1.0 # kreiss-oliger damping coefficient, max_step should be limited to avoid instability
    
    diss = np.zeros_like(current_state)
    for ivar in range(0, NUM_VARS) :
        ivar_values = current_state[(ivar)*N:(ivar+1)*N]
        ivar_diss = np.dot(my_grid.derivatives.d_dissipation_matrix, ivar_values) 
        diss[(ivar)*N:(ivar+1)*N] = ivar_diss
    rhs += sigma * diss
    #################################################################################################### 
    
    # see https://github.com/KAClough/BabyGRChombo/wiki/Useful-code-background
    
    # overwrite outer boundaries with extrapolation (order specified in uservariables.py)
    my_grid.fill_outer_boundary(rhs)

    # overwrite inner cells using parity under r -> - r
    my_grid.fill_inner_boundary(rhs)            
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
    
    #Finally return the rhs
    return rhs
