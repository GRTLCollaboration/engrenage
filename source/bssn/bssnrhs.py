# bssnrhs.py
# as in Etienne https://arxiv.org/abs/1712.07658v2
# see also Baumgarte https://arxiv.org/abs/1211.6632 for the eqns with matter

import numpy as np
from bssn.tensoralgebra import *

# phi is the (exponential) conformal factor, that is \gamma_ij = e^{4\phi) \bar gamma_ij
def get_bssn_rhs(bssn_rhs, r, bssn_vars, d1, d2, background, emtensor) :

    ####################################################################################################
    # Get all the useful quantities that will be used in the rhs
    
    em4phi = np.exp(-4.0*bssn_vars.phi)    
    bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
    bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)
    
    # The rescaled connections Delta^i, Delta^i_jk and Delta_ijk
    Delta_U, Delta_ULL, Delta_LLL  = get_tensor_connections(r, bssn_vars.h_LL, d1.h_LL, background)
    
    # \bar \Gamma^i_jk
    bar_chris = get_bar_christoffel(r, Delta_ULL, background)



   # rescaled shift in terms of scaling factors and bssn_vars.shift_U
    Shift_U = background.inverse_scaling_vector * bssn_vars.shift_U
    d1_Shift_U = (background.d1_inverse_scaling_vector * bssn_vars.shift_U[:,:,np.newaxis] 
                     + d1.shift_U * background.inverse_scaling_vector[:,:,np.newaxis])
    d2_Shift_U = (np.einsum('xijk,xi->xijk', background.d2_inverse_scaling_vector, bssn_vars.shift_U)
         + np.einsum('xik,xij->xijk', background.d1_inverse_scaling_vector, d1.shift_U)
         + np.einsum('xij,xik->xijk', background.d1_inverse_scaling_vector, d1.shift_U)
         + np.einsum('xi,xijk->xijk', background.inverse_scaling_vector, d2.shift_U)
    )

    # This is the conformal divergence of the shift \bar D_i \beta^i
    bar_div_shift =  np.einsum('xii->x', d1_Shift_U)
    bar_div_shift += np.einsum('xiij,xj->x', bar_chris, Shift_U)     

    # Trace of \bar Aij and A_ij A^ij
    trace_bar_A   = get_trace_bar_A(r, bssn_vars, background)       
    bar_A_squared = get_bar_A_squared(r, bssn_vars, background) 
    bar_A_LL = get_bar_A_LL(r, bssn_vars, background)
    bar_A_UU = get_bar_A_UU(r, bssn_vars, background)
    
    ####################################################################################################
    # First the conformal factor phi
    
    # Calculate rhs
    dphidt = (- one_sixth * bssn_vars.lapse * bssn_vars.K 
              + one_sixth * bar_div_shift)

    bssn_rhs.phi = dphidt     

    ####################################################################################################        
    # h is the rescaled part of the deviation from the hat metric
    # that is, \bar \gamma_ij = \hat \gamma_ij + \epsilon_ij
    # h_ij is rescaled \epsilon_ij (factors of 1/r etc)     
    
    # This is \hat\gamma_jk \hat D_i shift^k 
    # (note Etienne paper notation ambiguity - this is NOT \hat D_i \beta_j)
    hat_D_shift_U = (
         np.einsum('xjk,xki->xij', background.hat_gamma_LL, d1_Shift_U)
         + np.einsum('xjk,xkil,xl->xij', background.hat_gamma_LL, background.hat_christoffel, Shift_U)
   )
    
    # Rescale quantities because we want change in h not epsilon
    r_hat_D_shift_U = background.inverse_scaling_matrix * hat_D_shift_U
    r_bar_gamma_LL = get_rescaled_bar_gamma_LL(r, bssn_vars.h_LL, background)
    
    # Need to get the scalar factor in the right array dimension
    scalar_factor = two_thirds * (bssn_vars.lapse * trace_bar_A - bar_div_shift)
    
    # Now sum the values
    dhdt = (scalar_factor[:,np.newaxis,np.newaxis] * r_bar_gamma_LL
            - 2.0 * bssn_vars.lapse[:,np.newaxis,np.newaxis] * bssn_vars.a_LL 
            + r_hat_D_shift_U + np.transpose(r_hat_D_shift_U, axes=(0,2,1)))

    bssn_rhs.h_LL = dhdt     

    ####################################################################################################    
    # K is the trace of the extrinsic curvature 
    # that is K_ij = A_ij + 1/3 \gamma_ij K
    
    # Calculate \bar D^k \bar D_k lapse
    bar_D2_lapse = (np.einsum('xij,xij->x', bar_gamma_UU, d2.lapse)
                  - np.einsum('xij,xkij,xk->x', bar_gamma_UU, bar_chris, d1.lapse))

    # Calculate rhs    
    dKdt = (bssn_vars.lapse * (one_third * bssn_vars.K * bssn_vars.K 
                               + bar_A_squared + 0.5 * eight_pi_G * (emtensor.rho + emtensor.S))
            - em4phi * (bar_D2_lapse 
                        + 2.0 * np.einsum('xij,xi,xj->x', bar_gamma_UU, d1.lapse, d1.phi)))

    bssn_rhs.K = dKdt 

    ####################################################################################################    
    # a_ij is the rescaled version of the conformal, traceless part of the extrinsic curvature
    # that is A_ij =  e^{4\phi) \tilde A_ij
    # a_ij is rescaled \tilde A_ij (factors of 1/r etc)    
    
    # Ricci tensor
    bar_Rij = get_bar_ricci_tensor(r, bssn_vars.h_LL, d1.h_LL, d2.h_LL, bssn_vars.lambda_U, d1.lambda_U,
                                              Delta_U, Delta_ULL, Delta_LLL, 
                                              bar_gamma_UU, bar_gamma_LL, background)
    
    # \bar A_ik \bar A^k_j = gamma^kl A_ik A_jl
    AikAkj = np.einsum('xkl,xik,xlj->xij', bar_gamma_UU, bar_A_LL, bar_A_LL)
    
    # The trace free part of the evolution eqn for A_ij
    dAdt_TF_part = (bssn_vars.lapse[:,np.newaxis,np.newaxis] * 
                        (- 2.0 * d2.phi
                         + 4.0 * np.einsum('xi,xj->xij', d1.phi, d1.phi)
                         + 2.0 * np.einsum('xkij,xk->xij', bar_chris, d1.phi)
                         + bar_Rij - eight_pi_G * emtensor.Sij)
                      - d2.lapse
                      + np.einsum('xkij,xk->xij', bar_chris, d1.lapse)
                      + 2.0 * np.einsum('xi,xj->xij', d1.phi, d1.lapse)
                      + 2.0 * np.einsum('xj,xi->xij', d1.phi, d1.lapse))
    
    trace = get_trace(dAdt_TF_part, bar_gamma_UU)
    trace = trace[:,np.newaxis,np.newaxis]

    # Rescale quantities because we want change in a_ij not A_ij
    dadt_TF_part = background.inverse_scaling_matrix * dAdt_TF_part
    r_AikAkj = background.inverse_scaling_matrix * AikAkj
    r_bar_gamma_LL = get_rescaled_bar_gamma_LL(r, bssn_vars.h_LL, background)
    
    # Calculate rhs    
    dadt = ( - two_thirds * bar_div_shift[:,np.newaxis,np.newaxis] * bssn_vars.a_LL
             + bssn_vars.lapse[:,np.newaxis,np.newaxis] * (- 2.0 * r_AikAkj
                                                 + bssn_vars.K[:,np.newaxis,np.newaxis] * bssn_vars.a_LL)
             + em4phi[:,np.newaxis,np.newaxis] * (dadt_TF_part
                                                  - one_third * trace * r_bar_gamma_LL))

    bssn_rhs.a_LL = dadt    

    ####################################################################################################    
    # lambda^i is the rescaled version of the constrained quantity \Lambda^i = \Delta^i
    # Where \Delta^k = \bar\gamma^ij (\bar\Gamma^k_ij - \hat\Gamma^k_ij)  

    # \bar \gamma^jk \hat D_j \hat D_k shift^i
    hat_D2_shift = (  np.einsum('xjk,xijk->xi', bar_gamma_UU, d2_Shift_U)
                + np.einsum('xjk,xikl,xlj->xi', bar_gamma_UU, background.hat_christoffel, d1_Shift_U)
                + np.einsum('xjk,xijl,xlk->xi', bar_gamma_UU, background.hat_christoffel, d1_Shift_U)
                - np.einsum('xjk,xljk,xil->xi', bar_gamma_UU, background.hat_christoffel, d1_Shift_U)
                + np.einsum('xjk,xiklj,xl->xi', bar_gamma_UU, background.d1_hat_christoffel, Shift_U)
                + np.einsum('xjk,xijl,xlkm,xm->xi', bar_gamma_UU, background.hat_christoffel, 
                                                    background.hat_christoffel, Shift_U)
                - np.einsum('xjk,xljk,xilm,xm->xi', bar_gamma_UU, background.hat_christoffel, 
                                                    background.hat_christoffel, Shift_U))
    # This is \bar D^i (\bar D_j \beta^j) note the raised index of j
    # We can use that D_i V^i = 1/sqrt(detgamma) d_i [sqrt(detgamma) V^i]
    # And that we impose det(bargamma) = det(hatgamma) which we know the derivs for analytically
    bar_D_div_shift = (np.einsum('xij,xkjk->xi', bar_gamma_UU, d2_Shift_U)
                     + (0.5 / background.det_hat_gamma[:,np.newaxis] * 
                        np.einsum('xij,xkj,xk->xi', bar_gamma_UU, d1_Shift_U, background.d1_det_hat_gamma))
                     + (0.5 / background.det_hat_gamma[:,np.newaxis] * 
                        np.einsum('xij,xjk,xk->xi', bar_gamma_UU, background.d2_det_hat_gamma, Shift_U))
                     - (0.5 / background.det_hat_gamma[:,np.newaxis] / background.det_hat_gamma[:,np.newaxis] 
                        * np.einsum('xij,xj,xk,xk->xi', bar_gamma_UU, background.d1_det_hat_gamma, 
                                                        background.d1_det_hat_gamma, Shift_U)))

    # Calculate rhs
    dlambdadt = (hat_D2_shift 
                  + two_thirds * Delta_U * bar_div_shift[:,np.newaxis]
                  + one_third * bar_D_div_shift
                  - 2.0 * np.einsum('xij,xj->xi', bar_A_UU, d1.lapse)
                  + 12.0 * bssn_vars.lapse[:,np.newaxis] * np.einsum('xij,xj->xi', bar_A_UU, d1.phi)
                  + 2.0 * bssn_vars.lapse[:,np.newaxis] * np.einsum('xjk,xijk->xi', bar_A_UU, Delta_ULL)
                  - four_thirds * bssn_vars.lapse[:,np.newaxis] * np.einsum('xij,xj->xi', bar_gamma_UU, d1.K)
                  - 2.0 * eight_pi_G * bssn_vars.lapse[:,np.newaxis] * np.einsum('xij,xj->xi', bar_gamma_UU, emtensor.Si))
    
    # Rescale because we want change in lambda not Lambda
    dlambdadt[:] *= background.scaling_vector
    
    bssn_rhs.lambda_U = dlambdadt
    
    ####################################################################################################
    # end of bssn rhs
    ####################################################################################################