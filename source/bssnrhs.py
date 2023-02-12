# bssnrhs.py
# as in Etienne https://arxiv.org/abs/1712.07658v2
# see also Baumgarte https://arxiv.org/abs/1211.6632 for the eqns with matter

import numpy as np
from source.tensoralgebra import *

# phi is the (exponential) conformal factor, that is \gamma_ij = e^{4\phi) \bar gamma_ij
def get_rhs_phi(lapse, K, bar_div_shift) :

    # Calculate rhs
    dphidt = (- one_sixth * lapse * K 
              + one_sixth * bar_div_shift)
    
    return dphidt

# h is the rescaled part of the deviation from the flat conformal metric
# that is, \bar \gamma_ij = \hat \gamma_ij + \epsilon_ij
# h_ij is rescaled \epsilon_ij (factors of 1/r etc)
def get_rhs_h(r_here, r_gamma_LL, lapse, traceA, dshiftrdx, shiftr, bar_div_shift, a) :
    
    # This is \hat\gamma_jk \hat D_i shift^k (note Etienne paper notation ambiguity - this is not \hat D_i \beta_j)
    rflat_chris = get_rescaled_flat_spherical_chris(r_here)
    rhat_D_shift = np.zeros_like(rank_2_spatial_tensor)
    rhat_D_shift[i_r][i_r] = dshiftrdx
    rhat_D_shift[i_t][i_t] = rflat_chris[i_t][i_t][i_r] * shiftr
    rhat_D_shift[i_p][i_p] = rflat_chris[i_p][i_p][i_r] * shiftr

    # Calculate rhs
    dhdt = np.zeros_like(rank_2_spatial_tensor)
    for i in range(0, SPACEDIM):
        for j in range(0, SPACEDIM):
            
            # note that trace of \bar A_ij = 0 is enforced dynamically using the first term 
            # as in Etienne https://arxiv.org/abs/1712.07658v2 eqn (11a)
            # recall that h is the rescaled quantity so we need to scale all the terms
            dhdt[i][j] += ( two_thirds * r_gamma_LL[i][j] * (lapse * traceA - bar_div_shift)
                             + (rhat_D_shift[i][j] + rhat_D_shift[j][i])
                             - 2.0 * lapse * a[i][j])
    
    return dhdt

# K is the trace of the extrinsic curvature 
# that is K_ij = A_ij + 1/3 \gamma_ij K
def get_rhs_K(lapse, K, Asquared, em4phi, d2lapsedr2, dlapsedr, 
              r_conformal_chris, dphidr, r_gamma_UU, rho, S) :
    
    # Calculate \bar D^k \bar D_k lapse
    bar_D2_lapse = (r_gamma_UU[i_r][i_r] * (d2lapsedr2
                                               - r_conformal_chris[i_r][i_r][i_r] * dlapsedr)
                        - r_gamma_UU[i_t][i_t] * r_conformal_chris[i_r][i_t][i_t] * dlapsedr
                        - r_gamma_UU[i_p][i_p] * r_conformal_chris[i_r][i_p][i_p] * dlapsedr )

    # Calculate rhs    
    dKdt = (one_third * lapse * K * K 
            + lapse * Asquared 
            - em4phi * (bar_D2_lapse + 2.0 * r_gamma_UU[i_r][i_r] * dlapsedr * dphidr)
            + 0.5 * eight_pi_G * lapse * (rho + S))

    return dKdt

# a_ij is the rescaled version of the conformal, traceless part of the extrinsic curvature
# that is A_ij =  e^{4\phi) \tilde A_ij
# a_ij is rescaled \tilde A_ij (factors of 1/r etc)
def get_rhs_a(r_here, a, bar_div_shift, lapse, K, em4phi, rbar_Rij, 
              r_conformal_chris, r_gamma_UU, r_gamma_LL,
              d2phidr2, dphidr, d2lapsedr2, dlapsedr, h, dhdr, d2hdr2, rSij) : 
    
    #Some auxilliary quantities to be worked out first
    r_dAdt_TF_part    = np.zeros_like(rank_2_spatial_tensor)
    r_AikAkj          = np.zeros_like(rank_2_spatial_tensor)
    
    r_AikAkj[i_r][i_r] = a[i_r][i_r] * a[i_r][i_r] * r_gamma_UU[i_r][i_r]
    r_AikAkj[i_t][i_t] = a[i_t][i_t] * a[i_t][i_t] * r_gamma_UU[i_t][i_t]
    r_AikAkj[i_p][i_p] = a[i_p][i_p] * a[i_p][i_p] * r_gamma_UU[i_p][i_p]
    
    r_dAdt_TF_part[i_r][i_r] = (- 2.0 * lapse * d2phidr2 + 4.0 * lapse * dphidr * dphidr
                                                         + 4.0 * dlapsedr * dphidr
                                                         - d2lapsedr2)
    
    # Add the parts related to the flat chris with correct scaling, and the Delta term
    r_dAdt_TF_part[i_r][i_r] += (2.0 * lapse * dphidr + dlapsedr) * r_conformal_chris[i_r][i_r][i_r]
    r_dAdt_TF_part[i_t][i_t] =  (2.0 * lapse * dphidr + dlapsedr) * r_conformal_chris[i_r][i_t][i_t]
    r_dAdt_TF_part[i_p][i_p] =  (2.0 * lapse * dphidr + dlapsedr) * r_conformal_chris[i_r][i_p][i_p]
   
    for i in range(0, SPACEDIM):        
        for j in range(0, SPACEDIM):
            r_dAdt_TF_part[i][j] += (+ lapse * rbar_Rij[i][j]
                                     - lapse * eight_pi_G * rSij[i][j])
    
    r_fullgamma_UU = em4phi * r_gamma_UU    
    trace = get_trace(r_dAdt_TF_part, r_fullgamma_UU)

    # Calculate rhs     
    dadt = np.zeros_like(rank_2_spatial_tensor)    
    for i in range(0, SPACEDIM):
        for j in range(0, SPACEDIM):
            dadt[i][j] += ( - two_thirds * a[i][j] * bar_div_shift
                            - 2.0 * lapse * r_AikAkj[i][j] 
                            + lapse * a[i][j] * K
                            + em4phi * r_dAdt_TF_part[i][j]
                            - one_third * trace * (r_gamma_LL[i][j]) )
            
    return dadt

# lambda_r is the rescaled part of the constrained quantity \Lambda^i = \Delta^i
# Where \Delta^k = \bar\gamma^ij (\bar\Gamma^k_ij - \hat\Gamma^k_ij)
def get_rhs_lambdar(r_here, d2shiftrdr2, dshiftrdr, shiftr, h, dhdr, rDelta_U, rDelta_ULL, bar_div_shift,
                    r_gamma_UU, a_UU, lapse, dlapsedr, dphidr, dKdr, Si) :
    
    
    # Useful quantities
    rflat_chris = get_rescaled_flat_spherical_chris(r_here)
    
    # \bar \gamma^ij \hat D_i \hat D_j shift^r
    hat_D2_shiftr = (     r_gamma_UU[i_r][i_r] * d2shiftrdr2
                        - r_gamma_UU[i_t][i_t] * rflat_chris[i_r][i_t][i_t] * dshiftrdr
                        - r_gamma_UU[i_p][i_p] * rflat_chris[i_r][i_p][i_p] * dshiftrdr
                        + ( r_gamma_UU[i_t][i_t] * rflat_chris[i_r][i_t][i_t] 
                                                   * rflat_chris[i_t][i_r][i_t]  * shiftr)
                        + ( r_gamma_UU[i_p][i_p] * rflat_chris[i_r][i_p][i_p] 
                                                   * rflat_chris[i_p][i_r][i_p]  * shiftr) )

    # This is \bar D^r (\bar D_i \beta^i) note the raised index of r
    bar_D_div_shift = r_gamma_UU[i_r][i_r] * (d2shiftrdr2
                                                    + 2.0 / r_here * dshiftrdr 
                                                    - 2.0 / r_here / r_here * shiftr)
    
    # Calculate rhs
    dlambdardt = (hat_D2_shiftr 
                  + two_thirds * rDelta_U[i_r] * bar_div_shift
                  + one_third * bar_D_div_shift
                  - 2.0 * a_UU[i_r][i_r] * (dlapsedr 
                                                - 6.0 * lapse * dphidr
                                                - lapse * rDelta_ULL[i_r][i_r][i_r])
                  + 2.0 * a_UU[i_t][i_t] * lapse * rDelta_ULL[i_r][i_t][i_t]
                  + 2.0 * a_UU[i_p][i_p] * lapse * rDelta_ULL[i_r][i_p][i_p]
                  - four_thirds * lapse * r_gamma_UU[i_r][i_r] * dKdr
                  - 2.0 *  eight_pi_G * lapse * r_gamma_UU[i_r][i_r] * Si[i_r])
    
    return dlambdardt