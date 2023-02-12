#hamdiagnostics.py

# python modules
import numpy as np
import time

# homemade code
from source.uservariables import *
from source.fourthorderderivatives import *
from source.logderivatives import *
from source.gridfunctions import *
from source.tensoralgebra import *
from source.mymatter import *

# The diagnostic function returns the Hamiltonian constraint over the grid
# it takes in the solution of the evolution, which is the state vector at every
# time step, and returns the spatial profile Ham(r) at each time step
def get_Ham_diagnostic(solutions_over_time, t, R, N_r, r_is_logarithmic) :

    start = time.time()
    
    # Set up grid values
    dx, N, r, logarithmic_dr = setup_grid(R, N_r, r_is_logarithmic)
    
    # predefine some userful quantities
    oneoverlogdr = 1.0 / logarithmic_dr
    oneoverlogdr2 = oneoverlogdr * oneoverlogdr
    oneoverdx  = 1.0 / dx
    oneoverdxsquared = oneoverdx * oneoverdx
    
    Ham = []
    num_times = int(np.size(solutions_over_time) / (NUM_VARS * N))
    
    # unpack the vectors at each time
    for i in range(num_times) :
        
        t_i = t[i]
        
        if(num_times == 1):
            solution = solutions_over_time
        else :
            solution = solutions_over_time[i]

        # Unpack variables
        u, v , phi, hrr, htt, hpp, K, arr, att, app, lambdar, shiftr, br, lapse = unpack_state(solution, N_r)
        
        ################################################################################################

        # get the various derivs that we need in vector form
        # second derivatives
        d2udx2     = get_d2fdx2(u, oneoverdxsquared)
        d2phidx2   = get_d2fdx2(phi, oneoverdxsquared)
        d2hrrdx2   = get_d2fdx2(hrr, oneoverdxsquared)
        d2httdx2   = get_d2fdx2(htt, oneoverdxsquared)
        d2hppdx2   = get_d2fdx2(hpp, oneoverdxsquared)

        # first derivatives
        dudx       = get_dfdx(u, oneoverdx)
        dvdx       = get_dfdx(v, oneoverdx)
        dphidx     = get_dfdx(phi, oneoverdx)
        dhrrdx     = get_dfdx(hrr, oneoverdx)
        dhttdx     = get_dfdx(htt, oneoverdx)
        dhppdx     = get_dfdx(hpp, oneoverdx)
        darrdx     = get_dfdx(arr, oneoverdx)
        dattdx     = get_dfdx(att, oneoverdx)
        dappdx     = get_dfdx(app, oneoverdx)
        dKdx       = get_dfdx(K, oneoverdx)
        dlambdardx = get_dfdx(lambdar, oneoverdx)
        
        if(r_is_logarithmic) :
            
            #overwrite with logarithmic derivatives
            dudx       = get_logdfdx(u, oneoverlogdr)
            dvdx       = get_logdfdx(v, oneoverlogdr)
            dphidx     = get_logdfdx(phi, oneoverlogdr)
            dhrrdx     = get_logdfdx(hrr, oneoverlogdr)
            dhttdx     = get_logdfdx(htt, oneoverlogdr)
            dhppdx     = get_logdfdx(hpp, oneoverlogdr)
            darrdx     = get_logdfdx(arr, oneoverlogdr)
            dattdx     = get_logdfdx(att, oneoverlogdr)
            dappdx     = get_logdfdx(app, oneoverlogdr)
            dKdx       = get_logdfdx(K, oneoverlogdr)
            dlambdardx = get_logdfdx(lambdar, oneoverlogdr)
            # second derivatives
            d2udx2     = get_logd2fdx2(u, oneoverlogdr2)
            d2phidx2   = get_logd2fdx2(phi, oneoverlogdr2)
            d2hrrdx2   = get_logd2fdx2(hrr, oneoverlogdr2)
            d2httdx2   = get_logd2fdx2(htt, oneoverlogdr2)
            d2hppdx2   = get_logd2fdx2(hpp, oneoverlogdr2)
        
        #################################################################################################
    
        # make container for output values
        Ham_i   = np.zeros_like(u)
    
        #################################################################################################    
    
        # now iterate over the grid (vector) and calculate the diagnostic values
        for ix in range(num_ghosts, N-num_ghosts) :

            # where am I?
            r_here = r[ix]
        
            # Assign BSSN vars to local tensors
            h = np.zeros_like(rank_2_spatial_tensor)
            h[i_r][i_r] = hrr[ix]
            h[i_t][i_t] = htt[ix]
            h[i_p][i_p] = hpp[ix]
            em4phi = np.exp(-4.0*phi[ix])
        
            dhdr = np.zeros_like(rank_2_spatial_tensor)
            dhdr[i_r][i_r] = dhrrdx[ix]
            dhdr[i_t][i_t] = dhttdx[ix]
            dhdr[i_p][i_p] = dhppdx[ix]
        
            d2hdr2 = np.zeros_like(rank_2_spatial_tensor)
            d2hdr2[i_r][i_r] = d2hrrdx2[ix]
            d2hdr2[i_t][i_t] = d2httdx2[ix]
            d2hdr2[i_p][i_p] = d2hppdx2[ix]

            a = np.zeros_like(rank_2_spatial_tensor)
            a[i_r][i_r] = arr[ix]
            a[i_t][i_t] = att[ix]
            a[i_p][i_p] = app[ix]
        
            # Calculate some useful quantities
            ########################################################
        
            # \hat \Gamma^i_jk
            flat_chris = get_flat_spherical_chris(r_here)
        
            # rescaled \bar\gamma_ij
            r_gamma_LL = get_rescaled_metric(h)
            r_gamma_UU = get_rescaled_inverse_metric(h)
        
            # (unscaled) \bar\gamma_ij and \bar\gamma^ij
            bar_gamma_LL = get_metric(r_here, h)
            bar_gamma_UU = get_inverse_metric(r_here, h)
        
            # \bar A_ij, \bar A^ij and the trace A_i^i, then Asquared = \bar A_ij \bar A^ij
            bar_A_LL = get_A_LL(r_here, a)
            bar_A_UU = get_A_UU(a, r_gamma_UU, r_here)
            traceA   = get_trace_A(a, r_gamma_UU)
            Asquared = get_Asquared(a, r_gamma_UU)
        
            # The connections Delta^i, Delta^i_jk and Delta_ijk
            Delta_U, Delta_ULL, Delta_LLL  = get_connection(r_here, bar_gamma_UU, bar_gamma_LL, h, dhdr)
            bar_Rij = get_ricci_tensor(r_here, h, dhdr, d2hdr2, lambdar[ix], dlambdardx[ix], 
                                       Delta_U, Delta_ULL, Delta_LLL, bar_gamma_UU, bar_gamma_LL)
            bar_R   = get_trace(bar_Rij, bar_gamma_UU)
        
            # Matter sources
            matter_rho            = get_rho( u[ix], dudx[ix], v[ix], bar_gamma_UU, em4phi )

            # End of: Calculate some useful quantities, now start diagnostic
            #################################################################

            # Get the Ham constraint eqn (13) of Baumgarte https://arxiv.org/abs/1211.6632
            Ham_i[ix] = (  two_thirds * K[ix] * K[ix] - Asquared
                         + em4phi * ( bar_R
                                      - 8.0 * bar_gamma_UU[i_r][i_r] * (dphidx[ix] * dphidx[ix] 
                                                                        + d2phidx2[ix])
                                      # These terms come from \bar\Gamma^r d_r \phi from the \bar D^2 \phi term
                                      + 8.0 * bar_gamma_UU[i_t][i_t] * flat_chris[i_r][i_t][i_t] * dphidx[ix]
                                      + 8.0 * bar_gamma_UU[i_p][i_p] * flat_chris[i_r][i_p][i_p] * dphidx[ix]
                                      + 8.0 * Delta_U[i_r] * dphidx[ix])
                         - 2.0 * eight_pi_G * matter_rho )

        # end of iteration over grid 
        ###################################   
        
        # Add the Ham value to the output
        Ham.append(Ham_i)
        
    # end of iteration over time  
    #########################################################################
    
    end = time.time()
    #print("time at t= ", t_i, " is, ", end-start)
    
    return r, Ham
