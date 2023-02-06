#rhsevolution.py

# python modules
import numpy as np
import time

# homemade code
from source.uservariables import *
from source.fourthorderderivatives import *
from source.tensoralgebra import *
from source.mymatter import *
from source.bssn_rhs import *
    
# function that returns the rhs for each of the field vars
def get_rhs(t_i, vars_vec, R, N_r, eta, progress_bar, state) :

    # Some functions for timing and tracking progress
    start = time.time()
    
    # Set up grid values
    dx = R/N_r
    N = N_r + num_ghosts * 2 
    r = np.linspace(-(num_ghosts-0.5)*dx, R+(num_ghosts-0.5)*dx, N)
    
    # this is where the rhs will go
    rhs = np.zeros_like(vars_vec) 
    
    ####################################################################################################
    #unpackage the vector for readability - these are the vectors of values across r values at time t_i
    # see uservariables.py for naming conventions
    
    # Unpack variables from vars_vec - see uservariables.py
    u, v , phi, hrr, htt, hpp, K, arr, att, app, lambdar, shiftr, br, lapse = unpack_vars_vector(vars_vec, N_r)   

    ####################################################################################################
    # enforce that the determinant of \bar gamma_ij is equal to that of flat space in spherical coords
    # (note that trace of \bar A_ij = 0 is enforced dynamically below as in Etienne https://arxiv.org/abs/1712.07658v2)
    
    # iterate over the grid (vector)
    for ix in range(N) :

        # first the metric
        h = np.zeros_like(rank_2_spatial_tensor)
        h[i_r][i_r] = hrr[ix]
        h[i_t][i_t] = htt[ix]
        h[i_p][i_p] = hpp[ix]
        determinant = abs(get_rescaled_determinant_gamma(h))
        
        hrr[ix] = (1.0 + hrr[ix])/np.power(determinant,1./3) - 1.0
        htt[ix] = (1.0 + htt[ix])/np.power(determinant,1./3) - 1.0
        hpp[ix] = (1.0 + hpp[ix])/np.power(determinant,1./3) - 1.0
    
    ####################################################################################################

    # get the various derivs that we need to evolve things in vector form 
    # - see fourthorderderivatives.py
    
    oneoverdx  = 1.0 / dx
    oneoverdxsquared = oneoverdx * oneoverdx
    
    # second derivatives
    d2udx2     = get_d2fdx2(u, oneoverdxsquared)
    d2phidx2   = get_d2fdx2(phi, oneoverdxsquared)
    d2hrrdx2   = get_d2fdx2(hrr, oneoverdxsquared)
    d2httdx2   = get_d2fdx2(htt, oneoverdxsquared)
    d2hppdx2   = get_d2fdx2(hpp, oneoverdxsquared)
    d2lapsedx2 = get_d2fdx2(lapse, oneoverdxsquared)
    d2shiftrdx2 = get_d2fdx2(shiftr, oneoverdxsquared)

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
    dshiftrdx  = get_dfdx(shiftr, oneoverdx)
    dbrdx      = get_dfdx(br, oneoverdx)
    dlapsedx   = get_dfdx(lapse, oneoverdx)
    
    # first derivatives - advec left and right
    dudx_advec_L       = get_dfdx_advec_L(u, oneoverdx)
    dvdx_advec_L       = get_dfdx_advec_L(v, oneoverdx)
    dphidx_advec_L     = get_dfdx_advec_L(phi, oneoverdx)
    dhrrdx_advec_L     = get_dfdx_advec_L(hrr, oneoverdx)
    dhttdx_advec_L     = get_dfdx_advec_L(htt, oneoverdx)
    dhppdx_advec_L     = get_dfdx_advec_L(hpp, oneoverdx)
    darrdx_advec_L     = get_dfdx_advec_L(arr, oneoverdx)
    dattdx_advec_L     = get_dfdx_advec_L(att, oneoverdx)
    dappdx_advec_L     = get_dfdx_advec_L(app, oneoverdx)
    dKdx_advec_L       = get_dfdx_advec_L(K, oneoverdx)
    dlambdardx_advec_L = get_dfdx_advec_L(lambdar, oneoverdx)
    dshiftrdx_advec_L  = get_dfdx_advec_L(shiftr, oneoverdx)
    dbrdx_advec_L      = get_dfdx_advec_L(br, oneoverdx)
    dlapsedx_advec_L   = get_dfdx_advec_L(lapse, oneoverdx)
    
    dudx_advec_R       = get_dfdx_advec_R(u, oneoverdx)
    dvdx_advec_R       = get_dfdx_advec_R(v, oneoverdx)
    dphidx_advec_R     = get_dfdx_advec_R(phi, oneoverdx)
    dhrrdx_advec_R     = get_dfdx_advec_R(hrr, oneoverdx)
    dhttdx_advec_R     = get_dfdx_advec_R(htt, oneoverdx)
    dhppdx_advec_R     = get_dfdx_advec_R(hpp, oneoverdx)
    darrdx_advec_R     = get_dfdx_advec_R(arr, oneoverdx)
    dattdx_advec_R     = get_dfdx_advec_R(att, oneoverdx)
    dappdx_advec_R     = get_dfdx_advec_R(app, oneoverdx)
    dKdx_advec_R       = get_dfdx_advec_R(K, oneoverdx)
    dlambdardx_advec_R = get_dfdx_advec_R(lambdar, oneoverdx)
    dshiftrdx_advec_R  = get_dfdx_advec_R(shiftr, oneoverdx)
    dbrdx_advec_R      = get_dfdx_advec_R(br, oneoverdx)
    dlapsedx_advec_R   = get_dfdx_advec_R(lapse, oneoverdx)  
    
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
    # note that we do the ghost cells separately
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
        # (mostly from tensoralgebra.py)
        ########################################################
        
        # rescaled \bar\gamma_ij and \bar\gamma^ij
        r_gamma_LL = get_rescaled_metric(h)
        r_gamma_UU = get_rescaled_inverse_metric(h)
        
        # (unscaled) \bar\gamma_ij and \bar\gamma^ij
        bar_gamma_LL = get_metric(r_here, h)
        bar_gamma_UU = get_inverse_metric(r_here, h)
        
        # \bar A_ij, \bar A^ij and the trace A_i^i, then Asquared = \bar A_ij \bar A^ij
        a_UU = get_a_UU(a, r_gamma_UU)
        traceA   = get_trace_A(a, r_gamma_UU)
        Asquared = get_Asquared(a, r_gamma_UU)

        # The connections Delta^i, Delta^i_jk and Delta_ijk
        Delta_U, Delta_ULL, Delta_LLL  = get_connection(r_here, bar_gamma_UU, bar_gamma_LL, h, dhdr)
        rbar_Rij = get_rescaled_ricci_tensor(r_here, h, dhdr, d2hdr2, lambdar[ix], dlambdardx[ix], 
                                   r_gamma_UU, r_gamma_LL)        

        # \bar \gamma^i_jk
        conformal_chris = get_conformal_chris(Delta_ULL, r_here)

        # This is the conformal divergence of the shift \bar D_i \beta^i
        # Use the fact that the conformal metric determinant is \hat \gamma = r^4 sin2theta
        bar_div_shift =  (dshiftrdx[ix] + 2.0 * shiftr[ix] / r_here)
        
        # Matter sources - see mymatter.py
        matter_rho             = get_rho( u[ix], dudx[ix], v[ix], r_gamma_UU, em4phi )
        matter_Si              = get_Si(  u[ix], dudx[ix], v[ix])
        matter_S, matter_rSij  = get_rescaled_Sij( u[ix], dudx[ix], v[ix], r_gamma_UU, em4phi,
                                             r_gamma_LL)

        # End of: Calculate some useful quantities, now start RHS
        #########################################################

        # Get the matter rhs - see mymatter.py
        rhs_u[ix], rhs_v[ix] = get_matter_rhs(u[ix], v[ix], dudx[ix], d2udx2[ix], 
                                              bar_gamma_UU, em4phi, dphidx[ix], 
                                              K[ix], lapse[ix], dlapsedx[ix], conformal_chris)

        # Get the bssn rhs - see bssn.py
        rhs_phi[ix]     = get_rhs_phi(lapse[ix], K[ix], bar_div_shift)
        
        rhs_h           = get_rhs_h(r_here, r_gamma_LL, lapse[ix], traceA, dshiftrdx[ix], shiftr[ix], 
                                    bar_div_shift, a)
        
        rhs_K[ix]       = get_rhs_K(lapse[ix], K[ix], Asquared, em4phi, d2lapsedx2[ix], dlapsedx[ix], 
                                    conformal_chris, dphidx[ix], bar_gamma_UU, matter_rho, matter_S)
        
        rhs_a           = get_rhs_a(r_here, a, bar_div_shift, lapse[ix], K[ix], em4phi, rbar_Rij,
                                    conformal_chris, Delta_ULL, r_gamma_UU, bar_gamma_UU,
                                    d2phidx2[ix], dphidx[ix], d2lapsedx2[ix], dlapsedx[ix], 
                                    h, dhdr, d2hdr2, matter_rSij)
        
        rhs_lambdar[ix] = get_rhs_lambdar(r_here, d2shiftrdx2[ix], dshiftrdx[ix], shiftr[ix], h, dhdr,
                                          Delta_U, Delta_ULL, bar_div_shift,
                                          r_gamma_UU, a_UU, lapse[ix],
                                          dlapsedx[ix], dphidx[ix], dKdx[ix], matter_Si)
        
        # Set the gauge vars rhs
        # eta is the 1+log slicing damping coefficient - of order 1/M_adm of spacetime        
        rhs_br[ix]     = 0.75 * rhs_lambdar[ix] - eta * br[ix]
        rhs_shiftr[ix] = br[ix]
        rhs_lapse[ix]  = - 2.0 * lapse[ix] * K[ix]        
        
        # Add advection to time derivatives     
        if (shiftr[ix] > 0) :
            rhs_u[ix]       += shiftr[ix] * dudx_advec_R[ix]
            rhs_v[ix]       += shiftr[ix] * dvdx_advec_R[ix]
            rhs_phi[ix]     += shiftr[ix] * dphidx_advec_R[ix]
            rhs_hrr[ix]     = (rhs_h[i_r][i_r] + shiftr[ix] * dhrrdx_advec_R[ix] 
                               + 2.0 * hrr[ix] * dshiftrdx[ix])
            rhs_htt[ix]     = (rhs_h[i_t][i_t] + shiftr[ix] * dhttdx_advec_R[ix]
                               + 2.0 * shiftr[ix] * 1.0/r_here * h[i_t][i_t])
            rhs_hpp[ix]     = (rhs_h[i_p][i_p] + shiftr[ix] * dhppdx_advec_R[ix]
                               + 2.0 * shiftr[ix] * 1.0/r_here * h[i_p][i_p])
            rhs_K[ix]       += shiftr[ix] * dKdx_advec_R[ix]
            rhs_arr[ix]     = (rhs_a[i_r][i_r] + shiftr[ix] * darrdx_advec_R[ix] 
                               + 2.0 * arr[ix] * dshiftrdx[ix])
            rhs_att[ix]     = (rhs_a[i_t][i_t] + shiftr[ix] * dattdx_advec_R[ix]
                               + 2.0 * shiftr[ix] * 1.0/r_here * a[i_t][i_t])
            rhs_app[ix]     = (rhs_a[i_p][i_p] + shiftr[ix] * dappdx_advec_R[ix]
                               + 2.0 * shiftr[ix] * 1.0/r_here * a[i_p][i_p])
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
            rhs_hrr[ix]     = (rhs_h[i_r][i_r] + shiftr[ix] * dhrrdx_advec_L[ix]
                               + 2.0 * hrr[ix] * dshiftrdx[ix])
            rhs_htt[ix]     = (rhs_h[i_t][i_t] + shiftr[ix] * dhttdx_advec_L[ix]
                               + 2.0 * shiftr[ix] * 1.0/r_here * h[i_t][i_t])
            rhs_hpp[ix]     = (rhs_h[i_p][i_p] + shiftr[ix] * dhppdx_advec_L[ix]
                               + 2.0 * shiftr[ix] * 1.0/r_here * h[i_p][i_p])
            rhs_K[ix]       += shiftr[ix] * dKdx_advec_L[ix]
            rhs_arr[ix]     = (rhs_a[i_r][i_r] + shiftr[ix] * darrdx_advec_L[ix] 
                               + 2.0 * arr[ix] * dshiftrdx[ix])
            rhs_att[ix]     = (rhs_a[i_t][i_t] + shiftr[ix] * dattdx_advec_L[ix]
                               + 2.0 * shiftr[ix] * 1.0/r_here * a[i_t][i_t])
            rhs_app[ix]     = (rhs_a[i_p][i_p] + shiftr[ix] * dappdx_advec_L[ix]
                               + 2.0 * shiftr[ix] * 1.0/r_here * a[i_p][i_p])
            rhs_lambdar[ix] += (shiftr[ix] * dlambdardx_advec_L[ix] 
                                - lambdar[ix] * dshiftrdx[ix])
            # NB optional to add advection to lapse and shift vars
            # rhs_lapse       += shiftr[ix] * dlapsedx_advec_L[ix]            
            # rhs_br[ix]      += 0.0
            # rhs_shiftr[ix]  += 0.0
        
    # end of rhs iteration over grid points   
    ####################################################################################################        
        
    #package up the rhs values into a vector rhs (like vars_vec) for return - see uservariables.py
    pack_vars_vector(rhs, N_r, rhs_u, rhs_v , rhs_phi, rhs_hrr, rhs_htt, rhs_hpp, 
                     rhs_K, rhs_arr, rhs_att, rhs_app, rhs_lambdar, rhs_shiftr, rhs_br, rhs_lapse)

    #################################################################################################### 
            
    # finally add Kreiss Oliger dissipation
    sigma = 10.0 # kreiss-oliger damping coefficient, max_step should be limited to 0.1 R/N_r
    
    diss = np.zeros_like(vars_vec) 
    for ivar in range(0, NUM_VARS) :
        ivar_values = vars_vec[(ivar)*N:(ivar+1)*N]
        ivar_diss = get_dissipation(ivar_values, oneoverdx, sigma)
        diss[(ivar)*N:(ivar+1)*N] = ivar_diss
    rhs += diss
    
    #################################################################################################### 
        
    # overwrite outer boundaries with extrapolation (order specified in uservariables.py)
    for ivar in range(0, NUM_VARS) :
        boundary_cells = np.array([(ivar + 1)*N-3, (ivar + 1)*N-2, (ivar + 1)*N-1])
        var_asymptotic_power = asymptotic_power[ivar]
        for count, ix in enumerate(boundary_cells) :
            offset = -1 - count
            rhs[ix]    = rhs[ix + offset] * (r[N - 3 + count] / r[N - 4])**var_asymptotic_power

    # overwrite inner cells using parity under r -> - r
    for ivar in range(0, NUM_VARS) :
        boundary_cells = np.array([(ivar)*N, (ivar)*N+1, (ivar)*N+2])
        var_parity = parity[ivar]
        for count, ix in enumerate(boundary_cells) :
            offset = 5 - 2*count
            rhs[ix] = rhs[ix + offset] * var_parity           
                   
    #################################################################################################### 
    
    # Some code for checking timing and progress output
    end = time.time()
    #print("time at t= ", t_i, " is, ", end-start)
    
    # state is a list containing last updated time t:
    # state = [last_t, dt for progress bar]
    # its values can be carried between function calls throughout the ODE integration
    last_t, deltat = state
    
    # call update(n) here where n = (t - last_t) / dt
    n = int((t_i - last_t)/deltat)
    progress_bar.update(n)
    # we need this to take into account that n is a rounded number:
    state[0] = last_t + deltat * n
        
    ####################################################################################################
    
    #Finally return the rhs
    return rhs
