#tensoralgebra.py

# This file contains useful tensor algebra functions that are used in the RHS evolution
# and diagnostics etc

import numpy as np

SPACEDIM: int = 3

# Dividing quantities in advance usually reduces compute load
one_sixth = 1.0/6.0
one_third = 1.0/3.0
two_thirds = 2.0/3.0
four_thirds = 4.0/3.0

eight_pi_G = 8.0 * np.pi * 1.0 # Newtons constant, we take G=c=1

# Kronecker delta \delta_ij (i.e. identity matrix)
delta_ij = np.identity(SPACEDIM)

# struct for the emtensor, default to zero
class EMTensor :
    
    def __init__(self, N):
        
        self.rho = np.zeros([N])
        self.Sij = np.zeros([N,SPACEDIM, SPACEDIM])
        self.Si = np.zeros([N,SPACEDIM])
        self.S = np.zeros([N])

# Compute trace of some rank 2 tensor with indices lowered
# Assuming gamma_UU is the appropriate inverse metric
def get_trace(T_LL, gamma_UU):

    return np.einsum('xij,xij->x', gamma_UU, T_LL)

# Compute \bar A_ij given the rescaled perturbation a_ij
def get_bar_A_LL(r, bssn_vars, background):

    bar_A_LL = background.scaling_matrix * bssn_vars.a_LL
    
    return bar_A_LL

# Compute the \bar A^ij given a_ij and h_ij
def get_bar_A_UU(r, bssn_vars, background):
    
    bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)
    bar_A_LL = get_bar_A_LL(r, bssn_vars, background)
    bar_A_UU = (bar_gamma_UU @ bar_A_LL @ bar_gamma_UU)
    
    return bar_A_UU

# Compute a^ij given a_ij and h_ij
def get_a_UU(r, bssn_vars, background):
    
    r_bar_gamma_UU = get_rescaled_bar_gamma_UU(r, bssn_vars.h_LL, background)
    a_UU = r_bar_gamma_UU @ bssn_vars.a_LL @ r_bar_gamma_UU  
    
    return a_UU

# Compute trace of \bar A_ij
def get_trace_bar_A(r, bssn_vars, background) :

    r_bar_gamma_UU = get_rescaled_bar_gamma_UU(r, bssn_vars.h_LL, background)
    
    return get_trace(bssn_vars.a_LL, r_bar_gamma_UU)

# Compute \bar A_ij \bar A^ij
def get_bar_A_squared(r, bssn_vars, background) :
    
    r_bar_gamma_UU = get_rescaled_bar_gamma_UU(r, bssn_vars.h_LL, background)
    a_UU = r_bar_gamma_UU @ bssn_vars.a_LL @ r_bar_gamma_UU
    
    bar_A_squared = np.einsum('xij,xij->x', bssn_vars.a_LL, a_UU)

    return bar_A_squared

# Compute determinant of conformal spatial metric \bar\gamma
def get_det_bar_gamma(r, h_LL, background) :
    
    epsilon_LL = h_LL * background.scaling_matrix
    determinant = np.linalg.det(epsilon_LL + background.hat_gamma_LL)
    
    return determinant

# d1 det bar gamma / dx^i
# Uses that det bar gamma = det hat gamma
def get_d1_det_bar_gamma_dx(r, h_LL, background) :

    # asumming that initially det bar gamma = det hat gamma this should be preserved 
    # (we should always assume this, but it is not a requirement)
    
    return background.d1_det_hat_gamma 

# Computer the conformal spatial metric \bar \gamma_ij given the rescaled perturbation h
def get_bar_gamma_LL(r, h_LL, background) :

    epsilon_LL = h_LL * background.scaling_matrix
    bar_gamma_LL = epsilon_LL + background.hat_gamma_LL
    
    return bar_gamma_LL

# Computer the rescaled spatial metric n_ij * \bar \gamma_ij given the rescaled perturbation h
def get_rescaled_bar_gamma_LL(r, h_LL, background) :   
    
    r_bar_gamma_LL = h_LL + background.hat_gamma_LL * background.inverse_scaling_matrix
    
    return r_bar_gamma_LL

# Compute inverse of the conformal spatial metric \bar\gamma^ij given the rescaled perturbation h
def get_bar_gamma_UU(r, h_LL, background):

    bar_gamma_LL = get_bar_gamma_LL(r, h_LL, background)
    
    return np.linalg.inv(bar_gamma_LL)

def get_rescaled_bar_gamma_UU(r, h_LL, background):

    r_bar_gamma_LL = get_rescaled_bar_gamma_LL(r, h_LL, background)
    return  np.linalg.inv(r_bar_gamma_LL)

def get_vector_advection(r, V_U, advec_V_U, shift_U, d1_shift_U, background) :   
    
    advec_V_U = (np.einsum('xij,xj->xi', advec_V_U, background.inverse_scaling_vector * shift_U)
               + np.einsum('xij,xj->xi', background.scaling_vector[:,:,np.newaxis] * V_U[:,:,np.newaxis] 
                                         * background.d1_inverse_scaling_vector, background.inverse_scaling_vector * shift_U)
               - np.einsum('xij,xj->xi', d1_shift_U, background.inverse_scaling_vector * V_U)
               - np.einsum('xij,xj->xi', background.scaling_vector[:,:,np.newaxis] * shift_U[:,:,np.newaxis] 
                                         * background.d1_inverse_scaling_vector, background.inverse_scaling_vector * V_U))
    
    return advec_V_U

def get_tensor_advection(r, A_LL, advec_A_LL, shift_U, d1_shift_U, background) :    
    
    advec_A_LL = ( np.einsum('xijk,xk->xij', advec_A_LL, background.inverse_scaling_vector * shift_U)
                 + np.einsum('xijk,xk->xij', background.inverse_scaling_matrix[:,:,:,np.newaxis] * A_LL[:,:,:,np.newaxis] 
                                             * background.d1_scaling_matrix, background.inverse_scaling_vector * shift_U)
                 + np.einsum('xik,xkj->xij', A_LL, background.inverse_scaling_vector[:,np.newaxis,:] * d1_shift_U) 
                 + np.einsum('xjk,xki->xij', A_LL, background.inverse_scaling_vector[:,np.newaxis,:] * d1_shift_U)                   
                 + np.einsum('xik,xjk->xij', A_LL * background.scaling_vector[:,np.newaxis,:], 
                                             background.inverse_scaling_vector[:,:,np.newaxis] * shift_U[:,np.newaxis,:] 
                                             * background.d1_inverse_scaling_vector) 
                 + np.einsum('xjk,xik->xij', A_LL * background.scaling_vector[:,np.newaxis,:], 
                                             background.inverse_scaling_vector[:,:,np.newaxis] * shift_U[:,np.newaxis,:] 
                                             * background.d1_inverse_scaling_vector))
    
    return advec_A_LL

def get_bar_div_shift(r, bssn_vars, d1, background) :

    # The connections Delta^i, Delta^i_jk and Delta_ijk
    Delta_U, Delta_ULL, Delta_LLL  = get_tensor_connections(r, bssn_vars.h_LL, d1.h_LL, background)

    # \bar \Gamma^i_jk
    bar_chris = get_bar_christoffel(r, Delta_ULL, background)
        
    # This is the conformal divergence of the shift \bar D_i \beta^i
    bar_div_shift =  np.einsum('xii->x', d1.shift_U)
    bar_div_shift += np.einsum('xiij,xj->x', bar_chris, bssn_vars.shift_U)
    
    return bar_div_shift

def get_bar_christoffel(r, Delta_ULL, background) :

    return background.hat_christoffel + Delta_ULL

# See eqn (23)-(24) in Baumgarte https://arxiv.org/abs/1211.6632
# \Delta^i_{jk} \equiv \bar \Gamma^i_{jk} - \hat\Gamma^i_{jk}
# These quantities transform as tensors
def get_tensor_connections(r, h_LL, d1_h_dx, background) :

    # (unscaled) \bar\gamma_ij and \bar\gamma^ij
    bar_gamma_LL = get_bar_gamma_LL(r, h_LL, background)
    bar_gamma_UU = get_bar_gamma_UU(r, h_LL, background)
    
    # \hat{D} \bar{\gamma}_{ij} dx^k, note derivative index last
    hat_D_bar_gamma = get_hat_D_bar_gamma_LL(r, h_LL, d1_h_dx, background)
    
    Delta_ULL = (  0.5 * np.einsum('xil,xklj->xijk', bar_gamma_UU, hat_D_bar_gamma)
                 + 0.5 * np.einsum('xil,xjlk->xijk', bar_gamma_UU, hat_D_bar_gamma)
                 - 0.5 * np.einsum('xil,xjkl->xijk', bar_gamma_UU, hat_D_bar_gamma))
                            
    Delta_U = np.einsum('xjk,xijk->xi', bar_gamma_UU, Delta_ULL)           
    Delta_LLL = np.einsum('xil,xljk->xijk', bar_gamma_LL, Delta_ULL)
    
    return Delta_U, Delta_ULL, Delta_LLL

# Compute the \bar R_ij Ricci tensor
# See eqn (12) in Baumgarte https://arxiv.org/abs/1211.6632
def get_bar_ricci_tensor(r, h_LL, d1_h_dx, d2_h_dxdy, lambda_U, d1_lambda_dx, 
                         Delta_U, Delta_ULL, Delta_LLL, 
                         bar_gamma_UU, bar_gamma_LL, background) :
    
    # Get \hat D \bar \Lambda^i / dx^j
    hat_D_bar_Lambda_U = get_hat_D_bar_Lambda_U(r, lambda_U, d1_lambda_dx, background)
    # Get \bar\gamma^kl \hat D_k \hat D_l \bar\gamma_ij
    hat_D2_bar_gamma_LL = get_hat_D2_bar_gamma_LL(r, h_LL, d1_h_dx, d2_h_dxdy, background)
 
    bar_ricci = (- 0.5 * hat_D2_bar_gamma_LL
             + 0.5 * np.einsum('xki,xkj->xij', bar_gamma_LL, hat_D_bar_Lambda_U)
             + 0.5 * np.einsum('xkj,xki->xij', bar_gamma_LL, hat_D_bar_Lambda_U)
             + 0.5 * np.einsum('xk,xijk->xij', Delta_U, Delta_LLL)
             + 0.5 * np.einsum('xk,xjik->xij', Delta_U, Delta_LLL)
             + np.einsum('xkl,xmki,xjml->xij', bar_gamma_UU, Delta_ULL, Delta_LLL)
             + np.einsum('xkl,xmkj,ximl->xij', bar_gamma_UU, Delta_ULL, Delta_LLL)
             + np.einsum('xkl,xmik,xmjl->xij', bar_gamma_UU, Delta_ULL, Delta_LLL))
            
    return bar_ricci

# \hat D \bar \Lambda^i / dx^j derivative index last
# See eqn (26) in Baumgarte https://arxiv.org/abs/1211.6632
def get_hat_D_bar_Lambda_U(r, lambda_U, d1_lambda_dx, background) :

    N = np.size(r)
    Lambda_U = background.inverse_scaling_vector * lambda_U
    
    hat_D_Lambda = np.zeros([N, SPACEDIM, SPACEDIM])         
    
    hat_D_Lambda += (d1_lambda_dx * background.inverse_scaling_vector[:, :, np.newaxis]
                   + background.d1_inverse_scaling_vector * lambda_U[:, :, np.newaxis]
                   + np.einsum('xijk,xk->xij', background.hat_christoffel, Lambda_U))
                
    return hat_D_Lambda

# \bar|gamma^kl \hat D_k \hat D_l \bar\gamma_ij 
# See eqn (27) in Baumgarte https://arxiv.org/abs/1211.6632
def get_hat_D2_bar_gamma_LL(r, h_LL, d1_h_dx, d2_h_dxdy, background) :

    N = np.size(r)
    hat_D2_bar_gamma_LL = np.zeros([N, SPACEDIM, SPACEDIM])     
    
    # Useful quantities, and for readability
    bar_gamma_LL = get_bar_gamma_LL(r, h_LL, background)
    bar_gamma_UU = get_bar_gamma_UU(r, h_LL, background)
    hat_D_bar_gamma_LL_dx = get_hat_D_bar_gamma_LL(r, h_LL, d1_h_dx, background)
    hat_chris = background.hat_christoffel   
    d1_hat_chris_dx = background.d1_hat_christoffel
    d1_m_dx = background.d1_scaling_matrix
    d2_m_dxdy = background.d2_scaling_matrix
    epsilon_LL = h_LL * background.scaling_matrix
    
    d1_epsilon_LL_dx = (d1_h_dx * background.scaling_matrix[:,:,:,np.newaxis] + 
                        d1_m_dx * h_LL[:,:,:,np.newaxis])

    N = np.size(r)
    # This is dm_ij dx_k * dh_ij dx_l
    dm_dxk_dh_dxl = np.zeros([N, SPACEDIM, SPACEDIM, SPACEDIM, SPACEDIM])
    for k in np.arange(SPACEDIM) :
        for l in np.arange(SPACEDIM) :
            dm_dxk_dh_dxl[:,:,:,k,l] = d1_m_dx[:,:,:,k] * d1_h_dx[:,:,:,l]
    
    # First term in rhs of (27), bit of a faff
    hat_D2_bar_gamma_LL += ( np.einsum('xkl,xijkl->xij', bar_gamma_UU, d2_h_dxdy) * background.scaling_matrix
                           + np.einsum('xkl,xijkl->xij', bar_gamma_UU, d2_m_dxdy) * h_LL
                           + 2.0 * np.einsum('xkl,xijkl->xij', bar_gamma_UU, dm_dxk_dh_dxl)
                           - np.einsum('xkl,xmlik,xmj->xij', bar_gamma_UU, d1_hat_chris_dx, epsilon_LL)
                           - np.einsum('xkl,xmljk,xim->xij', bar_gamma_UU, d1_hat_chris_dx, epsilon_LL)
                           - np.einsum('xkl,xmli,xmjk->xij', bar_gamma_UU, hat_chris, d1_epsilon_LL_dx)
                           - np.einsum('xkl,xmlj,ximk->xij', bar_gamma_UU, hat_chris, d1_epsilon_LL_dx))
    
    # now add the christoffel terms
    hat_D2_bar_gamma_LL += (- np.einsum('xkl,xijm,xmlk->xij', bar_gamma_UU, hat_D_bar_gamma_LL_dx, hat_chris)
                            - np.einsum('xkl,xmjl,xmik->xij', bar_gamma_UU, hat_D_bar_gamma_LL_dx, hat_chris)
                            - np.einsum('xkl,ximl,xmjk->xij', bar_gamma_UU, hat_D_bar_gamma_LL_dx, hat_chris))
                                                                                  
    return hat_D2_bar_gamma_LL

# Covariant derivative of the spatial metric \hat{D} \bar{\gamma}_{ij} / dx^k with 
# respect to the hat metric
# See eqn (25) in Baumgarte https://arxiv.org/abs/1211.6632
def get_hat_D_bar_gamma_LL(r, h_LL, d1_h_dx, background) :

    N = np.size(r)
    hat_D_epsilon = np.zeros([N, SPACEDIM, SPACEDIM, SPACEDIM])   
    
    # This is s_{ij} * \partial_k h_{ij}
    hat_D_epsilon += d1_h_dx * background.scaling_matrix[:,:,:,np.newaxis]

    # This is h_{ij} * \partial_k s_{ij}
    hat_D_epsilon += background.d1_scaling_matrix * h_LL[:,:,:,np.newaxis]    
    
    # Add additional terms from christoffels
    epsilon_LL = h_LL * background.scaling_matrix
    
    hat_D_epsilon += - (  np.einsum('xlik,xlj->xijk', background.hat_christoffel, epsilon_LL)
                        + np.einsum('xljk,xil->xijk', background.hat_christoffel, epsilon_LL))
    
    return hat_D_epsilon
