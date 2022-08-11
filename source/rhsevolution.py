#rhsevolution.py

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
def get_rhs(vars_vec, t_i, p, q) :

    start = time.time()
    
    # this is where the rhs will go
    rhs = np.zeros_like(vars_vec) 
    
    ####################################################################################################
    #unpackage the vector for readability - these are the vectors of values across r values at time t_i
    # see uservariables.py for naming conventions
    
    # Unpack variables
    u, v , phi, hrr, htt, hpp, K, arr, att, app, lambdar, shiftr, br, lapse = unpack_vars_vector(vars_vec)   

    ####################################################################################################
    # enforce that the determinant of \bar gamma_ij is equal to that of flat space in spherical coords
    # (note that trace of \bar A_ij = 0 is enforced dynamically below as in Etienne https://arxiv.org/abs/1712.07658v2)

    shiftrL   = np.zeros_like(shiftr)
    
    # iterate over the grid (vector)
    for ix in range(N) :

        # first the metric
        h = np.zeros_like(rank_2_spatial_tensor)
        h[i_r][i_r] = hrr[ix]
        h[i_t][i_t] = htt[ix]
        h[i_p][i_p] = hpp[ix]
        determinant = get_rescaled_determinant_gamma(h)
        hrr[ix] = (1.0 + hrr[ix])/np.power(determinant,1./3) - 1.0
        htt[ix] = (1.0 + htt[ix])/np.power(determinant,1./3) - 1.0
        hpp[ix] = (1.0 + hpp[ix])/np.power(determinant,1./3) - 1.0
        
        # Also just for convenience work out the shift with lowered index
        shiftrL[ix] = shiftr[ix] * hrr[ix] * np.exp(4.0*phi[ix])
    
    ####################################################################################################

    # get the various derivs that we need to evolve things in vector form
    # second derivatives
    d2udx2     = get_d2fdx2(u)
    d2phidx2   = get_d2fdx2(phi)
    d2hrrdx2   = get_d2fdx2(hrr)
    d2httdx2   = get_d2fdx2(htt)
    d2hppdx2   = get_d2fdx2(hpp)
    d2lapsedx2 = get_d2fdx2(lapse)
    d2shiftrdx2 = get_d2fdx2(shiftr)

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
    dshiftrdx  = get_dfdx(shiftr)
    dshiftrLdx = get_dfdx(shiftrL)
    dbrdx      = get_dfdx(br)
    dlapsedx   = get_dfdx(lapse)
    
    ####################################################################################################
    
    # make containers for rhs values
    rhs_u   = np.zeros_like(u)
    rhs_v   = np.zeros_like(v)
    rhs_phi = np.zeros_like(phi)
    rhs_hrr = np.zeros_like(hrr)
    rhs_htt = np.zeros_like(htt)
    rhs_hpp = np.zeros_like(hpp)
    rhs_K   = np.zeros_like(K)
    rhs_arr = np.zeros_like(hrr)
    rhs_att = np.zeros_like(htt)
    rhs_app = np.zeros_like(hpp)    
    rhs_lambdar = np.zeros_like(lambdar)
    rhs_shiftr  = np.zeros_like(shiftr)
    rhs_br     = np.zeros_like(br)
    rhs_lapse   = np.zeros_like(lapse)
    
    ####################################################################################################    
    
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
        
        # rescaled \bar\gamma_ij and \bar\gamma^ij
        r_gamma_LL = get_rescaled_metric(h)
        r_gamma_UU = get_rescaled_inverse_metric(h)
        
        # (unscaled) \bar\gamma_ij and \bar\gamma^ij
        bar_gamma_LL = get_metric(r_here, h)
        bar_gamma_UU = get_inverse_metric(r_here, h)
        
        # \bar A_ij and its trace, \bar A_ij \bar A^ij
        bar_A_LL = get_A_LL(r_here, a)
        bar_A_UU = get_A_UU(a, r_gamma_UU)
        traceA   = get_trace_A(r_here, a, r_gamma_UU)
        Asquared = get_Asquared(r_here, a, r_gamma_UU)
        
        # This is the conformal divergence of the shift \bar D_i \beta^i
        # We use the fact that the determinant of the conformal metric is 
        # fixed to that of the flat space metric in spherical coords
        bar_div_shift = dshiftrdx[ix] + 2.0 / r_here * shiftr[ix]
        # This is D^r (\bar D_i \beta^i) note the raised index of r
        bar_D_div_shift = bar_gamma_UU[i_r][i_r] * (d2shiftrdx2[ix] 
                                                    + 2.0 / r_here * dshiftrdx[ix] 
                                                    - 2.0 / r_here / r_here * shiftr[ix])
        
        # Same trick for \bar D_k \bar D^k lapse
        bar_D2_lapse = bar_gamma_UU[i_r][i_r] * (d2lapsedx2[ix]
                                                + dlapsedx[ix] * dhrrdx[ix] * bar_gamma_UU[i_r][i_r]
                                                + 2.0 / r_here * dlapsedx[ix])
        
        # This is \hat D_i shift_j (note lowered indices)
        hat_D_shift = np.zeros_like(rank_2_spatial_tensor)
        hat_D_shift[i_r][i_r] = dshiftrLdx[ix]
        hat_D_shift[i_t][i_t] = - flat_chris[i_r][i_t][i_t] * shiftrL[ix]
        hat_D_shift[i_p][i_p] = - flat_chris[i_r][i_p][i_p] * shiftrL[ix]
        # \bar \gamma^ij \hat D_i \hat D_j shift^r
        hat_D2_shift = bar_gamma_UU[i_r][i_r] * d2shiftrdx2[ix]
                
        # The connections Delta^i, Delta^i_jk and Delta_ijk
        Delta_U, Delta_ULL, Delta_LLL  = get_connection(r_here, bar_gamma_UU, bar_gamma_LL, h, dhdr)
        bar_Rij = get_ricci_tensor(r_here, h, dhdr, d2hdr2, lambdar[ix], dlambdardx[ix], 
                                   Delta_U, Delta_ULL, Delta_LLL, bar_gamma_UU, bar_gamma_LL)
        
        # Matter sources
        matter_rho            = get_rho( u[ix], dudx[ix], v[ix], bar_gamma_UU, em4phi )
        matter_Si             = get_Si(  u[ix], dudx[ix], v[ix], bar_gamma_UU, em4phi )
        matter_S, matter_Sij  = get_Sij( u[ix], dudx[ix], v[ix], bar_gamma_UU, em4phi,
                                             bar_gamma_LL)

        # End of: Calculate some useful quantities, now start RHS
        #########################################################

        # Get the matter rhs
        rhs_u[ix], rhs_v[ix] = get_matter_rhs(u[ix], v[ix], dudx[ix], d2udx2[ix], 
                                              bar_gamma_UU, em4phi, dphidx[ix], K[ix], lapse[ix], dlapsedx[ix])

        # Add advection
        rhs_u[ix] += shiftr[ix] * dudx[ix]
        rhs_v[ix] += shiftr[ix] * dvdx[ix]
        
        # USEFUL DEBUG: check flat space result
        #rhs_u[ix] = v[ix]
        #rhs_v[ix] = d2udx2[ix]

        # Get the bssn rhs - see bssn.py
        rhs_phi[ix]     = get_rhs_phi(lapse[ix], K[ix], bar_div_shift)
        
        rhs_h           = get_rhs_h(r_here, r_gamma_LL, lapse[ix], traceA, bar_div_shift, 
                                    hat_D_shift, a)
        
        rhs_K[ix]       = get_rhs_K(lapse[ix], K[ix], Asquared, em4phi, bar_D2_lapse, 
                                    dlapsedx[ix], dphidx[ix], bar_gamma_UU, matter_rho, matter_S)       
        
        rhs_a           = get_rhs_a(a, bar_div_shift, lapse[ix], K[ix], em4phi, bar_Rij,
                                    r_here, Delta_ULL, bar_gamma_UU, bar_A_UU, bar_A_LL,
                                    d2phidx2[ix], dphidx[ix], d2lapsedx2[ix], dlapsedx[ix], 
                                    h, dhdr, d2hdr2, matter_Sij)
        
        rhs_lambdar[ix] = get_rhs_lambdar(hat_D2_shift, Delta_U, Delta_ULL, bar_div_shift, 
                                          bar_D_div_shift, bar_gamma_UU, bar_A_UU, lapse[ix], dlapsedx[ix], 
                                          dphidx[ix], dKdx[ix], matter_Si)
        
        # Add advection to time derivatives
        rhs_phi[ix]     += shiftr[ix] * dphidx[ix]
        rhs_hrr[ix]     = rhs_h[i_r][i_r] + shiftr[ix] * dhrrdx[ix] - 2.0 * hrr[ix] * dshiftrdx[ix]
        rhs_htt[ix]     = rhs_h[i_t][i_t] + shiftr[ix] * dhttdx[ix]      
        rhs_hpp[ix]     = rhs_h[i_p][i_p] + shiftr[ix] * dhppdx[ix]       
        rhs_K[ix]       += shiftr[ix] * dKdx[ix]
        rhs_arr[ix]     = rhs_a[i_r][i_r] + shiftr[ix] * darrdx[ix] - 2.0 * arr[ix] * dshiftrdx[ix] 
        rhs_att[ix]     = rhs_a[i_t][i_t] + shiftr[ix] * dattdx[ix]        
        rhs_app[ix]     = rhs_a[i_p][i_p] + shiftr[ix] * dappdx[ix] 
        rhs_lambdar[ix] += shiftr[ix] * dlambdardx[ix] - lambdar[ix] * dshiftrdx[ix]       

        # Set the gauge vars rhs
        rhs_br[ix]     = 0.75 * rhs_lambdar[ix] - eta * br[ix]
        rhs_shiftr[ix] = br[ix]
        rhs_lapse[ix]  = - 2.0 * lapse[ix] * K[ix] + shiftr[ix] * dlapsedx[ix]

        
    # end of rhs iteration over grid points   
    ####################################################################################################        
        
    #package up the rhs values into a vector like vars_vec for return 
  
    # Scalar field vars
    rhs[idx_u * N : (idx_u + 1) * N] = rhs_u
    rhs[idx_v * N : (idx_v + 1) * N] = rhs_v
    
    # Conformal factor and rescaled perturbation to spatial metric
    rhs[idx_phi * N : (idx_phi + 1) * N] = rhs_phi
    rhs[idx_hrr * N : (idx_hrr + 1) * N] = rhs_hrr
    rhs[idx_htt * N : (idx_htt + 1) * N] = rhs_htt
    rhs[idx_hpp * N : (idx_hpp + 1) * N] = rhs_hpp

    # Mean curvature and rescaled perturbation to traceless A_ij
    rhs[idx_K   * N : (idx_K   + 1) * N] = rhs_K
    rhs[idx_arr * N : (idx_arr + 1) * N] = rhs_arr
    rhs[idx_att * N : (idx_att + 1) * N] = rhs_att
    rhs[idx_app * N : (idx_app + 1) * N] = rhs_app

    # Gamma^x, shift and lapse
    rhs[idx_lambdar * N : (idx_lambdar + 1) * N] = rhs_lambdar
    rhs[idx_shiftr  * N : (idx_shiftr  + 1) * N] = rhs_shiftr
    rhs[idx_br      * N : (idx_br      + 1) * N] = rhs_br
    rhs[idx_lapse   * N : (idx_lapse   + 1) * N] = rhs_lapse

    #################################################################################################### 
            
    # finally add Kreiss Oliger dissipation
    diss = np.zeros_like(vars_vec) 
    for ivar in range(0, NUM_VARS) :
        diss[(ivar-1)*N:ivar*N] = get_dissipation(vars_vec[(ivar-1)*N:ivar*N])
    rhs += diss
    
    #################################################################################################### 
        
    # overwrite outer boundaries with extrapolation (zeroth order)
    for ivar in range(0, NUM_VARS) :
        boundary_cells = np.array([(ivar + 1)*N-3, (ivar + 1)*N-2, (ivar + 1)*N-1])
        for count, ix in enumerate(boundary_cells) :
            offset = -1 - count
            rhs[ix]    = rhs[ix + offset]

    # overwrite inner cells using parity under r -> - r
    for ivar in range(0, NUM_VARS) :
        boundary_cells = np.array([(ivar)*N, (ivar)*N+1, (ivar)*N+2])
        var_parity = parity[ivar]
        for count, ix in enumerate(boundary_cells) :
            offset = 5 - 2*count
            rhs[ix] = rhs[ix + offset] * var_parity           
                   
    #################################################################################################### 
    
    #print("time at t= ", t_i, " , ", rhs_u, rhs_v)
    
    end = time.time()
    #print("time at t= ", t_i, " is, ", end-start)
    
    return rhs
