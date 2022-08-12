#diagnostics.py

# python modules
import numpy as np
import time

# homemade code
from myparams import *
from source.uservariables import *
from source.fourthorderderivatives import *
from source.tensoralgebra import *
from source.mymatter import *
from source.bssn_rhs import *

# function that returns the rhs for each of the field vars
# Assumes 3 ghost cells at either end of the vector of values

# klein gordon eqn
def get_diagnostics(solution) :

    start = time.time()
    
    Ham = []
    
    # unpack the vectors at each time t_i
    for i, t_i in enumerate(t) : 

        # Scalar field vars
        u    = solution[i, idx_u * N : (idx_u + 1) * N]
        v    = solution[i, idx_v * N : (idx_v + 1) * N]
    
        # Conformal factor and rescaled perturbation to spatial metric
        phi    = solution[i, idx_phi * N : (idx_phi + 1) * N]
        hrr    = solution[i, idx_hrr * N : (idx_hrr + 1) * N]
        htt    = solution[i, idx_htt * N : (idx_htt + 1) * N]
        hpp    = solution[i, idx_hpp * N : (idx_hpp + 1) * N]

        # Mean curvature and rescaled perturbation to traceless A_ij
        K      = solution[i, idx_K   * N : (idx_K   + 1) * N]
        arr    = solution[i, idx_arr * N : (idx_arr + 1) * N]
        att    = solution[i, idx_att * N : (idx_att + 1) * N]
        app    = solution[i, idx_app * N : (idx_app + 1) * N]

        # Gamma^x, shift and lapse
        lambdar    = solution[i, idx_lambdar * N : (idx_lambdar + 1) * N]
        shiftr     = solution[i, idx_shiftr  * N : (idx_shiftr  + 1) * N]
        br         = solution[i, idx_br      * N : (idx_br      + 1) * N]
        lapse      = solution[i, idx_lapse   * N : (idx_lapse   + 1) * N]    
    
        ################################################################################################

        # get the various derivs that we need to evolve things in vector form
        # second derivatives
        d2udx2     = get_d2fdx2(u)
        d2phidx2   = get_d2fdx2(phi)
        d2hrrdx2   = get_d2fdx2(hrr)
        d2httdx2   = get_d2fdx2(htt)
        d2hppdx2   = get_d2fdx2(hpp)

        # first derivatives
        dudx       = get_dfdx(u)
        dvdx       = get_dfdx(v)
        dphidx     = get_dfdx(phi)
        dhrrdx     = get_dfdx(hrr)
        dhttdx     = get_dfdx(htt)
        dhppdx     = get_dfdx(hpp)
        darrdx     = get_dfdx(arr)
        dattdx     = get_dfdx(att)
        dappdx     = get_dfdx(app)
        dKdx       = get_dfdx(K)
        dlambdardx = get_dfdx(lambdar)
    
        #################################################################################################
    
        # make containers for rhs values
        Ham_i   = np.zeros_like(u)
    
        #################################################################################################    
    
        # now iterate over the grid (vector) and calculate the rhs values
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
            bar_A_UU = get_A_UU(a, r_gamma_UU)
            traceA   = get_trace_A(a, r_gamma_UU)
            Asquared = get_Asquared(a, r_gamma_UU)
        
            # The connections Delta^i, Delta^i_jk and Delta_ijk
            Delta_U, Delta_ULL, Delta_LLL  = get_connection(r_here, bar_gamma_UU, bar_gamma_LL, h, dhdr)
            bar_Rij = get_ricci_tensor(r_here, h, dhdr, d2hdr2, lambdar[ix], dlambdardx[ix], 
                                       Delta_U, Delta_ULL, Delta_LLL, bar_gamma_UU, bar_gamma_LL)
            bar_R   = get_trace(bar_Rij, bar_gamma_UU)
        
            # Matter sources
            matter_rho            = get_rho( u[ix], dudx[ix], v[ix], bar_gamma_UU, em4phi )
            matter_Si             = get_Si(  u[ix], dudx[ix], v[ix], bar_gamma_UU, em4phi )
            matter_S, matter_Sij  = get_Sij( u[ix], dudx[ix], v[ix], bar_gamma_UU, em4phi,
                                             bar_gamma_LL)

            # End of: Calculate some useful quantities, now start diagnostic
            #################################################################

            # Get the Ham constraint eqn (13) of Baumgarte https://arxiv.org/abs/1211.6632
            Ham_i[ix] = (  two_thirds * K[ix] * K[ix] - Asquared
                         + em4phi * ( bar_R
                                      - 8.0 * bar_gamma_UU[i_r][i_r] * (dphidx[ix] * dphidx[ix] 
                                                                        + d2phidx2[ix])
                                      + 8.0 * bar_gamma_UU[i_t][i_t] * flat_chris[i_r][i_t][i_t] * dphidx[ix] 
                                      + 8.0 * bar_gamma_UU[i_p][i_p] * flat_chris[i_r][i_p][i_p] * dphidx[ix]
                                      + 8.0 * Delta_U[i_r] * dphidx[ix])
                         - 2.0 * eight_pi_G * matter_rho )
            
            #print("bar_R is ", bar_R)

        # end of iteration over grid 
        ###################################   
        
        # Add the Ham value to the output
        Ham.append(Ham_i)
        
    # end of iteration over time  
    #########################################################################
    
    end = time.time()
    #print("time at t= ", t_i, " is, ", end-start)
    
    return Ham
