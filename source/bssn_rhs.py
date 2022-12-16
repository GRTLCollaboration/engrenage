# bssn_rhs.py
# as in Etienne https://arxiv.org/abs/1712.07658v2
# see also Baumgarte https://arxiv.org/abs/1211.6632 for the eqns with matter

import numpy as np
from myparams import *
from source.tensoralgebra import *

def get_rhs_phi(lapse, K, bar_div_shift) :
    
    dphidt = (- one_sixth * lapse * K 
              + one_sixth * bar_div_shift)
    
    return dphidt

def get_rhs_h(r_here, r_gamma_LL, lapse, traceA, bar_div_shift, hat_D_shift, a) :
    
    dhdt = np.zeros_like(rank_2_spatial_tensor)
    inv_scaling = np.array([1.0, 1.0/r_here , 1.0/r_here/sintheta])  
    for i in range(0, SPACEDIM):
        for j in range(0, SPACEDIM):
            
            # note that trace of \bar A_ij = 0 is enforced dynamically using the first term 
            # as in Etienne https://arxiv.org/abs/1712.07658v2 eqn (11a)
            # recall that h is the rescaled quantity so we need to scale
            # Also note that there is a typo in Etienne
            dhdt[i][j] += ( two_thirds * r_gamma_LL[i][j] * (lapse * traceA - bar_div_shift)
                             + inv_scaling[i] * inv_scaling[j] * (hat_D_shift[i][j] + hat_D_shift[j][i])
                             - 2.0 * lapse * a[i][j])
    
    return dhdt

def get_rhs_K(lapse, K, Asquared, em4phi, bar_D2_lapse, dlapsedr, dphidr, bar_gamma_UU, rho, S) :
    
    dKdt = (one_third * lapse * K * K 
            + lapse * Asquared 
            - em4phi * (bar_D2_lapse + 2.0 * bar_gamma_UU[i_r][i_r] * dlapsedr * dphidr)
            + 0.5 * eight_pi_G * lapse * (rho + S))

    return dKdt

def get_rhs_a(a, bar_div_shift, lapse, K, em4phi, bar_Rij, r_here, Delta_ULL,
              bar_gamma_UU, bar_A_UU, bar_A_LL,
              d2phidr2, dphidr, d2lapsedr2, dlapsedr, h, dhdr, d2hdr2, Sij) : 
    
    #Some auxilliary quantities to be worked out first
    inv_scaling     = np.array([1.0, 1.0/r_here , 1.0/r_here/sintheta])
    dAdt_TF_part    = np.zeros_like(rank_2_spatial_tensor)
    r_AikAkj        = np.zeros_like(rank_2_spatial_tensor)
    chris           = get_conformal_chris(Delta_ULL, r_here)
    
    for i in range(0, SPACEDIM):        
        for j in range(0, SPACEDIM):
            dAdt_TF_part[i][j] += (delta[i][i_r] * delta[j][i_r] * (- 2.0 * lapse * d2phidr2
                                                                    + 4.0 * lapse * dphidr * dphidr
                                                                    + 4.0 * dlapsedr * dphidr
                                                                    - d2lapsedr2)
                                    + lapse * bar_Rij[i][j]
                                    - lapse * eight_pi_G * Sij[i][j])
            
            for k in range(0, SPACEDIM):
                dAdt_TF_part[i][j] += delta[k][i_r] * chris[k][i][j] * (+ 2.0 * lapse * dphidr 
                                                                        + dlapsedr)
                
                for l in range(0, SPACEDIM): 
                    r_AikAkj[i][j] += a[i][k] * bar_A_LL[j][l] * bar_gamma_UU[k][l]
    
    gamma_UU = em4phi * bar_gamma_UU    
    trace = get_trace(dAdt_TF_part, gamma_UU)
       
    dadt = np.zeros_like(rank_2_spatial_tensor)    
    for i in range(0, SPACEDIM):
        for j in range(0, SPACEDIM):
            dadt[i][j] += ( - two_thirds * a[i][j] * bar_div_shift
                            - 2.0 * lapse * r_AikAkj[i][j] 
                            + lapse * a[i][j] * K
                            + em4phi * dAdt_TF_part[i][j] * inv_scaling[i] * inv_scaling[j]
                            - one_third * trace * (delta[i][j] + h[i][j]) )
            
    return dadt

def get_rhs_lambdar(hat_D2_shiftr, Delta_U, Delta_ULL, bar_div_shift, bar_D_div_shift, 
                    bar_gamma_UU, bar_A_UU, lapse, dlapsedr, dphidr, dKdr, Si) :
    
    dlambdardt = (hat_D2_shiftr 
                  + two_thirds * Delta_U[i_r] * bar_div_shift
                  + one_third * bar_D_div_shift
                  - 2.0 * bar_A_UU[i_r][i_r] * (dlapsedr 
                                                - 6.0 * lapse * dphidr
                                                - lapse * Delta_ULL[i_r][i_r][i_r])
                  + 2.0 * bar_A_UU[i_t][i_t] * lapse * Delta_ULL[i_r][i_t][i_t]
                  + 2.0 * bar_A_UU[i_p][i_p] * lapse * Delta_ULL[i_r][i_p][i_p]
                  - four_thirds * lapse * bar_gamma_UU[i_r][i_r] * dKdr
                  - 2.0 *  eight_pi_G * lapse * bar_gamma_UU[i_r][i_r] * Si[i_r])
                  
    for i in range(0, SPACEDIM):
        for j in range(0, SPACEDIM):
            dlambdardt += 2.0 * lapse * bar_A_UU[i][j] * Delta_ULL[i_r][i][j]
    
    return dlambdardt


