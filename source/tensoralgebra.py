#tensoralgebra.py

# This file contains useful tensor algebra functions that are used in the RHS evolution
# and diagnostics etc

import numpy as np

# The flat spherical coordinate quantities as in arXiv:1211.6632
# Coordinate indices - r theta and phi
i_r = 0
i_t = 1
i_p = 2

# Spherical symmetry for now
sintheta = 1
sin2theta = 1
costheta = 0
cos2theta = 0

# Dividing quantities in advance usually reduces compute load
one_sixth = 1.0/6.0
one_third = 1.0/3.0
two_thirds = 2.0/3.0
four_thirds = 4.0/3.0

# Assume 3 spatial dimensions and set up structures for tensors
SPACEDIM = 3
eight_pi_G = 8.0 * np.pi * 1.0 # Newtons constant, we take G=c=1
rank_1_spatial_tensor = np.zeros(SPACEDIM)
rank_2_spatial_tensor = np.array([rank_1_spatial_tensor, rank_1_spatial_tensor, rank_1_spatial_tensor])
rank_3_spatial_tensor = np.array([rank_2_spatial_tensor, rank_2_spatial_tensor, rank_2_spatial_tensor])

# Kronecker delta \delta_ij (i.e. identity matrix)
delta = np.identity(SPACEDIM)

# flat spherical christoffel symbols
# See eqn (18) in Baumgarte https://arxiv.org/abs/1211.6632
def get_flat_spherical_chris(r_here) :

    spherical_chris = np.zeros_like(rank_3_spatial_tensor)
    one_over_r = 1.0 / r_here
    
    # non zero r comps \Gamma^r_ab
    spherical_chris[i_r][i_t][i_t] = - r_here
    spherical_chris[i_r][i_p][i_p] = - r_here * sin2theta
    
    # non zero theta comps \Gamma^theta_ab
    spherical_chris[i_t][i_p][i_p] = - sintheta * costheta 
    spherical_chris[i_t][i_r][i_t] = one_over_r
    spherical_chris[i_t][i_t][i_r] = one_over_r

    # non zero theta comps \Gamma^phi_ab
    spherical_chris[i_p][i_p][i_r] = one_over_r
    spherical_chris[i_p][i_r][i_p] = one_over_r
    spherical_chris[i_p][i_t][i_p] = costheta / sintheta
    spherical_chris[i_p][i_p][i_t] = costheta / sintheta
    
    return spherical_chris

# flat spherical christoffel symbols rescaled for r^2 factors 
# (as if they were tensors) as in Baumgarte https://arxiv.org/abs/1211.6632
def get_rescaled_flat_spherical_chris(r_here) :

    spherical_chris = np.zeros_like(rank_3_spatial_tensor)
    one_over_r = 1.0 / r_here
    
    # non zero r comps \Gamma^r_ab
    spherical_chris[i_r][i_t][i_t] = - one_over_r
    spherical_chris[i_r][i_p][i_p] = - one_over_r
    
    # non zero theta comps \Gamma^theta_ab
    spherical_chris[i_t][i_p][i_p] = - costheta * one_over_r 
    spherical_chris[i_t][i_r][i_t] = one_over_r
    spherical_chris[i_t][i_t][i_r] = one_over_r

    # non zero theta comps \Gamma^phi_ab
    spherical_chris[i_p][i_p][i_r] = one_over_r
    spherical_chris[i_p][i_r][i_p] = one_over_r
    spherical_chris[i_p][i_t][i_p] = costheta / sintheta * one_over_r
    spherical_chris[i_p][i_p][i_t] = costheta / sintheta * one_over_r
    
    return spherical_chris

def get_conformal_chris(Delta_ULL, r_here) :

    return get_flat_spherical_chris(r_here) + Delta_ULL

def get_rescaled_conformal_chris(rDelta_ULL, r_here) :

    return get_rescaled_flat_spherical_chris(r_here) + rDelta_ULL
    
# Compute determinant of conformal spatial metric \bar\gamma divided by r^4 sin^2 theta
def get_rescaled_determinant_gamma(h) :
    
    determinant = (1.0 + h[i_r][i_r]) * (1.0 + h[i_t][i_t]) * (1.0 + h[i_p][i_p])
    
    return determinant

# Computer the conformal spatial metric \bar \gamma_ij given the rescaled perturbation h
def get_metric(r_here, h) :
    
    scaling = np.array([1.0, r_here , r_here*sintheta])    

    gamma_LL = np.zeros_like(rank_2_spatial_tensor)   

    gamma_LL[i_r][i_r] =   scaling[i_r] * scaling[i_r] * ( 1.0 + h[i_r][i_r] )    
    gamma_LL[i_t][i_t] =   scaling[i_t] * scaling[i_t] * ( 1.0 + h[i_t][i_t] )     
    gamma_LL[i_p][i_p] =   scaling[i_p] * scaling[i_p] * ( 1.0 + h[i_p][i_p] )
    
    return gamma_LL

# Computer the rescaled spatial metric inv_scaling_ij * \bar \gamma_ij given the rescaled perturbation h
def get_rescaled_metric(h) :   
    
    r_gamma_LL = np.zeros_like(rank_2_spatial_tensor)   

    r_gamma_LL[i_r][i_r] =   ( 1.0 + h[i_r][i_r] )    
    r_gamma_LL[i_t][i_t] =   ( 1.0 + h[i_t][i_t] )    
    r_gamma_LL[i_p][i_p] =   ( 1.0 + h[i_p][i_p] )
    
    return r_gamma_LL

# Computer inverse of the conformal spatial metric \bar\gamma^ij given the rescaled perturbation h
def get_inverse_metric(r_here, h):
    
    inv_scaling = np.array([1.0, 1.0/r_here , 1.0/r_here/sintheta])

    # We assume the metric perturbation h is diagonal.
    # Just act on the diagonal, then promote back to a 3x3 matrix
    # Note that np.diag both promotes and demotes!
    diag_h = np.diag(h)
    return np.diag(inv_scaling * inv_scaling / (1. + diag_h))

def get_rescaled_inverse_metric(h):
    
    # We assume the metric perturbation h is diagonal.
    # Just act on the diagonal, then promote back to a 3x3 matrix
    # Note that np.diag both promotes and demotes!
    diag_h = np.diag(h)
    return np.diag( 1. / (1. + diag_h ) )

# Compute the \bar A_ij given the rescaled perturbation a
def get_A_LL(r_here, a):

    scaling = np.array([1.0, r_here , r_here*sintheta])    

    # We assume that `a` is diagonal.
    # Just act on the diagonal, then promote back to a 3x3 matrix
    # Note that np.diag both promotes and demotes!
    return np.diag( scaling * scaling * np.diag(a) )

# Compute the \bar A^ij given A_ij and \bar\gamma^ij
def get_A_UU(a, r_gamma_UU, r_here):

    inv_scaling = np.array([1.0, 1.0/r_here , 1.0/r_here/sintheta])
    
    # Note that bar_gamma_UU is symmetric.
    # If it wasn't symmetric, we would need to be careful about
    # transposing when multiplying from the right.
    # multi_dot is kind of overkill here for 3x3 matrices...
    # but it's good to know about!
    a_UU = np.linalg.multi_dot( [ r_gamma_UU, a, r_gamma_UU ] ) 
    
    return np.diag( inv_scaling * inv_scaling * np.diag(a_UU) )

# Compute a^ij given a_ij
def get_a_UU(a, r_gamma_UU):
    
    # Note that bar_gamma_UU is symmetric.
    # If it wasn't symmetric, we would need to be careful about
    # transposing when multiplying from the right.
    # multi_dot is kind of overkill here for 3x3 matrices...
    # but it's good to know about!
    a_UU = np.linalg.multi_dot( [ r_gamma_UU, a, r_gamma_UU ] ) 
    
    return a_UU

# Compute trace of (traceless part of) extrinsic curvature
def get_trace_A(a, r_gamma_UU) :

    # Matrix multiply, then matrix trace
    return np.trace( np.dot(r_gamma_UU, a ) )

# Compute trace of some rank 2 tensor with indices lowered
def get_trace(T_LL, gamma_UU):

    # Matrix multiply, then matrix trace
    return np.trace( np.dot(gamma_UU, T_LL ) )

# Compute A_ij A^ij
def get_Asquared(a, r_gamma_UU) :
    
    Asquared = 0.0
    for i in range(0, SPACEDIM):
        for j in range(0, SPACEDIM):    
            for k in range(0, SPACEDIM):
                for l in range(0, SPACEDIM):  
                    Asquared += ( a[i][j] *  a[k][l]
                                 * r_gamma_UU[i][k] * r_gamma_UU[j][l])

    return Asquared

# Compute connection of the spatial metric
# See eqn (23)-(24) in Baumgarte https://arxiv.org/abs/1211.6632
# \Delta^i_{jk} \equiv \bar \Gamma^i_{jk} - \Gamma0^i_{jk}
#               = (1/2) * \bar \gamma^{il} ( hat D_j \bar \gamma_{kl} + hat D_k \bar \gamma_{jl} - hat D_l \bar \gamma_{jk} )
#               = (1/2) * \bar \gamma^{il} ( hat D_j \epsilon_{kl} + hat D_k \epsilon_{jl} - hat D_l \epsilon_{jk} )
# where hat D_i is the covariant derivative associated with the flat metric \eta_{ij}.
def get_connection(r_here, bar_gamma_UU, bar_gamma_LL, h, dhdr) :
    
    Delta_ULL = np.zeros_like(rank_3_spatial_tensor)
    Delta_LLL = np.zeros_like(rank_3_spatial_tensor)
    hat_D_bar_gamma = get_hat_D_bar_gamma(r_here, h, dhdr)
    
    for i in range(0, SPACEDIM):
        for j in range(0, SPACEDIM):    
            for k in range(0, SPACEDIM):
                for m in range(0, SPACEDIM):
                    Delta_ULL[i][j][k] += 0.5 * bar_gamma_UU[i][m] * (  hat_D_bar_gamma[j][k][m] 
                                                                      + hat_D_bar_gamma[k][j][m] 
                                                                      - hat_D_bar_gamma[m][j][k])
                    
    Delta_U =  np.zeros_like(rank_1_spatial_tensor)           
    for i in range(0, SPACEDIM):
        for j in range(0, SPACEDIM):    
            for k in range(0, SPACEDIM):
                Delta_U[k] += bar_gamma_UU[i][j] * Delta_ULL[k][i][j]
                
                for m in range(0, SPACEDIM):
                    Delta_LLL[i][j][k] += bar_gamma_LL[i][m] * Delta_ULL[m][j][k]
    
    return Delta_U, Delta_ULL, Delta_LLL

# Compute rescaled connection of the spatial metric
# See eqn (23)-(24) in Baumgarte https://arxiv.org/abs/1211.6632
# \Delta^i_{jk} \equiv \bar \Gamma^i_{jk} - \Gamma0^i_{jk}
#               = (1/2) * \bar \gamma^{il} ( hat D_j \bar \gamma_{kl} + hat D_k \bar \gamma_{jl} - hat D_l \bar \gamma_{jk} )
#               = (1/2) * \bar \gamma^{il} ( hat D_j \epsilon_{kl} + hat D_k \epsilon_{jl} - hat D_l \epsilon_{jk} )
# where hat D_i is the covariant derivative associated with the flat metric \eta_{ij}.
def get_rescaled_connection(r_here, r_gamma_UU, r_gamma_LL, h, dhdr) :
    
    rDelta_ULL = np.zeros_like(rank_3_spatial_tensor)
    rDelta_LLL = np.zeros_like(rank_3_spatial_tensor)
    rhat_D_bar_gamma = get_rescaled_hat_D_bar_gamma(r_here, h, dhdr)
    
    for i in range(0, SPACEDIM):
        for j in range(0, SPACEDIM):    
            for k in range(0, SPACEDIM):
                for m in range(0, SPACEDIM):
                    rDelta_ULL[i][j][k] += 0.5 * r_gamma_UU[i][m] * (  rhat_D_bar_gamma[j][k][m] 
                                                                     + rhat_D_bar_gamma[k][j][m] 
                                                                     - rhat_D_bar_gamma[m][j][k])
                    
    rDelta_U =  np.zeros_like(rank_1_spatial_tensor)           
    for i in range(0, SPACEDIM):
        for j in range(0, SPACEDIM):    
            for k in range(0, SPACEDIM):
                rDelta_U[k] += r_gamma_UU[i][j] * rDelta_ULL[k][i][j]
                
                for m in range(0, SPACEDIM):
                    rDelta_LLL[i][j][k] += r_gamma_LL[i][m] * rDelta_ULL[m][j][k]
    
    return rDelta_U, rDelta_ULL, rDelta_LLL

# Compute the Ricci tensor
# See eqn (12) in Baumgarte https://arxiv.org/abs/1211.6632
def get_ricci_tensor(r_here, h, dhdr, d2hdr2, lambdar, dlambardr, 
                     Delta_U, Delta_ULL, Delta_LLL, 
                     bar_gamma_UU, bar_gamma_LL) :

    ricci = np.zeros_like(rank_2_spatial_tensor)
    
    # Get \hat D \Lambda^i
    hat_D_Lambda = get_hat_D_Lambda(r_here, lambdar, dlambardr)
    # Get \bar\gamma^kl \hat D_k \hat D_l \bar\gamma_ij
    hat_D2_bar_gamma = get_hat_D2_bar_gamma(r_here, h, dhdr, d2hdr2, bar_gamma_UU)
    
    for i in range(0, SPACEDIM):
        for j in range(0, SPACEDIM):
            ricci[i][j] += -0.5 * hat_D2_bar_gamma[i][j]
                
            for k in range(0, SPACEDIM):
                ricci[i][j] += (   0.5 * (bar_gamma_LL[k][i] * hat_D_Lambda[j][k] + 
                                          bar_gamma_LL[k][j] * hat_D_Lambda[i][k])
                                 + 0.5 * (Delta_U[k] * Delta_LLL[i][j][k] + 
                                          Delta_U[k] * Delta_LLL[j][i][k]) )
                
                for l in range(0, SPACEDIM):
                    for m in range(0, SPACEDIM): 
                        ricci[i][j] += bar_gamma_UU[k][l] * (  Delta_ULL[m][k][i] * 
                                                               Delta_LLL[j][m][l]
                                                             + Delta_ULL[m][k][j] * 
                                                               Delta_LLL[i][m][l]
                                                             + Delta_ULL[m][i][k] * 
                                                               Delta_LLL[m][j][l] )
            
    return ricci

# Compute the rescaled Ricci tensor
# See eqn (12) in Baumgarte https://arxiv.org/abs/1211.6632
def get_rescaled_ricci_tensor(r_here, h, dhdr, d2hdr2, lambdar, dlambardr,
                              rDelta_U, rDelta_ULL, rDelta_LLL, 
                              r_gamma_UU, r_gamma_LL) :

    r_ricci = np.zeros_like(rank_2_spatial_tensor)
    
    # Get \hat D \Lambda^i
    rhat_D_Lambda = get_rescaled_hat_D_Lambda(r_here, lambdar, dlambardr)
    # Get \bar\gamma^kl \hat D_k \hat D_l \bar\gamma_ij
    rhat_D2_bar_gamma = get_rescaled_hat_D2_bar_gamma(r_here, h, dhdr, d2hdr2, r_gamma_UU)
    
    for i in range(0, SPACEDIM):
        for j in range(0, SPACEDIM):
            r_ricci[i][j] += -0.5 * rhat_D2_bar_gamma[i][j]
                
            for k in range(0, SPACEDIM):
                r_ricci[i][j] += (   0.5 * (r_gamma_LL[k][i] * rhat_D_Lambda[j][k] + 
                                            r_gamma_LL[k][j] * rhat_D_Lambda[i][k])
                                   + 0.5 * (rDelta_U[k] * rDelta_LLL[i][j][k]  + 
                                            rDelta_U[k] * rDelta_LLL[j][i][k]) )
                
                for l in range(0, SPACEDIM):
                    for m in range(0, SPACEDIM): 
                        r_ricci[i][j] += r_gamma_UU[k][l] * (  rDelta_ULL[m][k][i] * 
                                                               rDelta_LLL[j][m][l]
                                                             + rDelta_ULL[m][k][j] * 
                                                               rDelta_LLL[i][m][l]
                                                             + rDelta_ULL[m][i][k] * 
                                                               rDelta_LLL[m][j][l] )
            
    return r_ricci

# \hat D_i \Lambda^j
# See eqn (26) in Baumgarte https://arxiv.org/abs/1211.6632
def get_hat_D_Lambda(r_here, lambdar, dlambardr) :
    
    # Make an array for \hat D_i \Lambda^j
    hat_D_Lambda = np.zeros_like(rank_2_spatial_tensor)
    
    # Useful quantities
    flat_chris = get_flat_spherical_chris(r_here)

    hat_D_Lambda[i_r][i_r] = dlambardr
    for i in range(0, SPACEDIM):
        for j in range(0, SPACEDIM):
            hat_D_Lambda[i][j] += flat_chris[j][i][i_r] * lambdar
                
    return hat_D_Lambda

# \hat D_i \Lambda^j
# See eqn (26) in Baumgarte https://arxiv.org/abs/1211.6632
def get_rescaled_hat_D_Lambda(r_here, lambdar, dlambardr) :
    
    # Make an array for \hat D_i \Lambda^j
    rhat_D_Lambda = np.zeros_like(rank_2_spatial_tensor)
    
    # Useful quantities
    rflat_chris = get_rescaled_flat_spherical_chris(r_here)

    rhat_D_Lambda[i_r][i_r] = dlambardr
    for i in range(0, SPACEDIM):
        for j in range(0, SPACEDIM):
            rhat_D_Lambda[i][j] += rflat_chris[j][i][i_r] * lambdar
                
    return rhat_D_Lambda

# \bar|gamma^kl \hat D_k \hat D_l \bar\gamma_ij 
# See eqn (27) in Baumgarte https://arxiv.org/abs/1211.6632
def get_hat_D2_bar_gamma(r_here, h, dhdr, d2hdr2, bar_gamma_UU) :
    
    hat_D2_bar_gamma = np.zeros_like(rank_2_spatial_tensor)

    # Useful quantities
    hat_D_bar_gamma = get_hat_D_bar_gamma(r_here, h, dhdr)
    flat_chris = get_flat_spherical_chris(r_here)
    r2 = r_here * r_here
    r2sin2theta = r2 * sin2theta
    
    # explicitly add non zero terms in spherical symmetry
    hat_D2_bar_gamma[i_r][i_r] = bar_gamma_UU[i_r][i_r] *   d2hdr2[i_r][i_r]
    
    hat_D2_bar_gamma[i_t][i_t] = bar_gamma_UU[i_r][i_r] * ( d2hdr2[i_t][i_t] * r2
                                                            + dhdr[i_t][i_t] * 2.0 * r_here )
    hat_D2_bar_gamma[i_p][i_p] = bar_gamma_UU[i_r][i_r] * ( d2hdr2[i_p][i_p] * r2sin2theta 
                                                            + dhdr[i_p][i_p] * 2.0 * r_here * sin2theta )
    
    # now add the christoffel terms
    for i in range(0, SPACEDIM):
        for j in range(0, SPACEDIM):            
            for k in range(0, SPACEDIM): 
                for l in range(0, SPACEDIM):  
                    for m in range(0, SPACEDIM): 
                        hat_D2_bar_gamma[i][j] += - bar_gamma_UU[k][l] * (  hat_D_bar_gamma[m][i][j] * flat_chris[m][l][k]
                                                                          + hat_D_bar_gamma[l][m][j] * flat_chris[m][i][k]
                                                                          + hat_D_bar_gamma[l][i][m] * flat_chris[m][j][k] )
                                                                        
                
    return hat_D2_bar_gamma

# \bar|gamma^kl \hat D_k \hat D_l \bar\gamma_ij 
# See eqn (27) in Baumgarte https://arxiv.org/abs/1211.6632
def get_rescaled_hat_D2_bar_gamma(r_here, h, dhdr, d2hdr2, r_gamma_UU) :
    
    hat_D2_bar_gamma = np.zeros_like(rank_2_spatial_tensor)

    # Useful quantities
    one_over_r = 1.0 / r_here
    hat_D_bar_gamma = get_rescaled_hat_D_bar_gamma(r_here, h, dhdr)
    rflat_chris = get_rescaled_flat_spherical_chris(r_here)
    
    # explicitly add non zero terms in spherical symmetry
    hat_D2_bar_gamma[i_r][i_r] = r_gamma_UU[i_r][i_r] *   d2hdr2[i_r][i_r]
    
    hat_D2_bar_gamma[i_t][i_t] = r_gamma_UU[i_r][i_r] * ( d2hdr2[i_t][i_t]
                                                            + dhdr[i_t][i_t] * 2.0 * one_over_r )
    hat_D2_bar_gamma[i_p][i_p] = r_gamma_UU[i_r][i_r] * ( d2hdr2[i_p][i_p]
                                                            + dhdr[i_p][i_p] * 2.0 * one_over_r )

    # now add the christoffel terms
    for i in range(0, SPACEDIM):
        for j in range(0, SPACEDIM): 
            hat_D2_bar_gamma[i][j] += (r_gamma_UU[i_t][i_t] * hat_D_bar_gamma[i_r][i][j] * one_over_r +
                                       r_gamma_UU[i_p][i_p] * hat_D_bar_gamma[i_r][i][j] * one_over_r * sin2theta)
    
    hat_D2_bar_gamma[i_r][i_r] += - 2.0 * (r_gamma_UU[i_t][i_t] * hat_D_bar_gamma[i_t][i_t][i_r] * one_over_r
                                           + r_gamma_UU[i_p][i_p] * hat_D_bar_gamma[i_p][i_p][i_r] * one_over_r)
    
    hat_D2_bar_gamma[i_t][i_t] += - 2.0 * (r_gamma_UU[i_r][i_r] * hat_D_bar_gamma[i_r][i_t][i_t] * one_over_r
                                             - r_gamma_UU[i_t][i_t] * hat_D_bar_gamma[i_t][i_r][i_t] * one_over_r)
    
    hat_D2_bar_gamma[i_p][i_p] += - 2.0 * (r_gamma_UU[i_r][i_r] * hat_D_bar_gamma[i_r][i_p][i_p] * one_over_r
                                             - r_gamma_UU[i_p][i_p] * hat_D_bar_gamma[i_p][i_r][i_p] * one_over_r * sin2theta)
    
    
    return hat_D2_bar_gamma

# We split the conformal metric into the form
# \bar \gamma_{ij} = \eta_{ij} + \e_{ij}
# where \eta_{ij} is the flat metric (in spherical polar coordinates), and where we write 
# \e_{ij} in the form
#                 /  h_rr                  r h_rt                  r sin(theta) h_rp     \
# \e_{ij} =       |  r h_rt                r^2 h_tt                r^2 sin(theta) h_tp   |
#                 \  r sin(theta) h_rp     r^2 sin(theta) h_tp     r^2 sin^2(theta) h_pp /
# here h is the rescaled perturbation to the flat metric

# Covariant derivative of the spatial metric \hat{D} \bar{\gamma}_{ij} with 
# respect to the flat metric in spherical polar coordinates
# See eqn (25) in Baumgarte https://arxiv.org/abs/1211.6632
def get_hat_D_bar_gamma(r_here, h, dhdr) :

    hat_D_epsilon = np.zeros_like(rank_3_spatial_tensor)
    
    # assume spherical symmetry
    dhdtheta = np.zeros_like(dhdr)
    dhdphi = np.zeros_like(dhdr)
    
    # some useful quantities
    r2 = r_here * r_here
    scaling = np.array([1.0, r_here , r_here*sintheta])
    
    # Fill derivatives \hat D_k epsilon_ij
    for i in range(0, SPACEDIM):
        for j in range(0, SPACEDIM):
            hat_D_epsilon[i_r, i, j]   = dhdr[i][j]     * scaling[i] * scaling[j]
            hat_D_epsilon[i_t, i, j]   = dhdtheta[i][j] * scaling[i] * scaling[j]
            hat_D_epsilon[i_p, i, j]   = dhdphi[i][j]   * scaling[i] * scaling[j]
            
    # Add additional terms from christoffels etc
    
    # d/dtheta
    hat_D_epsilon[i_t, i_r, i_r ] += - 2.0 *      h[i_r][i_t]
    hat_D_epsilon[i_t, i_t, i_t ] +=   2.0 * r2 * h[i_r][i_t]
    hat_D_epsilon[i_t, i_p, i_p ] +=   0.0
    
    hat_D_epsilon[i_t, i_r, i_t ] += scaling[i_r] * scaling[i_t] * (  h[i_r][i_r] - h[i_t][i_t] )
    hat_D_epsilon[i_t, i_t, i_r ] = hat_D_epsilon[i_t, i_r, i_t] 
    
    hat_D_epsilon[i_t, i_r, i_p ] += scaling[i_r] * scaling[i_p] * (- h[i_t][i_p])
    hat_D_epsilon[i_t, i_p, i_r ] = hat_D_epsilon[i_t, i_r, i_p]

    hat_D_epsilon[i_t, i_t, i_p ] += scaling[i_t] * scaling[i_p] * (  h[i_r][i_p])
    hat_D_epsilon[i_t, i_p, i_t ] = hat_D_epsilon[i_t, i_t, i_p]
    
    # d/dphi
    hat_D_epsilon[i_p, i_r, i_r ] += - 2.0 *      sintheta * h[i_r][i_p]
    hat_D_epsilon[i_p, i_t, i_t ] += - 2.0 * r2 * costheta * h[i_t][i_p]
    hat_D_epsilon[i_p, i_p, i_p ] +=   2.0 * r2 * sin2theta * ( costheta * h[i_t][i_p] 
                                                              + sintheta * h[i_r][i_p] )
    
    hat_D_epsilon[i_p, i_r, i_t ] += scaling[i_r] * scaling[i_t] * (- costheta * h[i_r][i_p] 
                                                                    - sintheta * h[i_t][i_p] )
    hat_D_epsilon[i_p, i_t, i_r ] = hat_D_epsilon[i_p, i_r, i_t] 
    
    hat_D_epsilon[i_p, i_r, i_p ] += scaling[i_r] * scaling[i_p] * (  costheta * h[i_r][i_t]
                                                                    + sintheta * h[i_r][i_r]
                                                                    - sintheta * h[i_p][i_p] )
    hat_D_epsilon[i_p, i_p, i_r ] = hat_D_epsilon[i_p, i_r, i_p]

    hat_D_epsilon[i_p, i_t, i_p ] += scaling[i_t] * scaling[i_p] * ( sintheta * h[i_r][i_t]                                                                                  + costheta * h[i_t][i_t]
                                                                   - costheta * h[i_p][i_p] )
    hat_D_epsilon[i_p, i_p, i_t ] = hat_D_epsilon[i_p, i_t, i_p]
            
    return hat_D_epsilon

# Covariant derivative of the spatial metric \hat{D} \bar{\gamma}_{ij} with 
# respect to the flat metric in spherical polar coordinates
# (Here we simplified for spherical symmetry, unlike above in the un-rescaled case which is general)
# See eqn (25) in Baumgarte https://arxiv.org/abs/1211.6632
def get_rescaled_hat_D_bar_gamma(r_here, h, dhdr) :

    hat_D_epsilon = np.zeros_like(rank_3_spatial_tensor)
    
    # assume spherical symmetry
    dhdtheta = np.zeros_like(dhdr)
    dhdphi = np.zeros_like(dhdr)
    
    # some useful quantities
    r2 = r_here * r_here
    scaling = np.array([1.0, r_here , r_here*sintheta])
    
    # Fill derivatives \hat D_k epsilon_ij
    for i in range(0, SPACEDIM):
        for j in range(0, SPACEDIM):
            hat_D_epsilon[i_r, i, j]   = dhdr[i][j]
            hat_D_epsilon[i_t, i, j]   = 0.0
            hat_D_epsilon[i_p, i, j]   = 0.0
            
    # Add additional terms from christoffels etc
    
    # d/dtheta
    hat_D_epsilon[i_t, i_r, i_r ] +=   0.0
    hat_D_epsilon[i_t, i_t, i_t ] +=   0.0
    hat_D_epsilon[i_t, i_p, i_p ] +=   0.0
    
    hat_D_epsilon[i_t, i_r, i_t ] += (  h[i_r][i_r] - h[i_t][i_t] ) / scaling[i_t]
    hat_D_epsilon[i_t, i_t, i_r ] = hat_D_epsilon[i_t, i_r, i_t] 
    
    hat_D_epsilon[i_t, i_r, i_p ] += 0.0
    hat_D_epsilon[i_t, i_p, i_r ] = hat_D_epsilon[i_t, i_r, i_p]

    hat_D_epsilon[i_t, i_t, i_p ] += 0.0
    hat_D_epsilon[i_t, i_p, i_t ] = hat_D_epsilon[i_t, i_t, i_p]
    
    # d/dphi
    hat_D_epsilon[i_p, i_r, i_r ] += 0.0
    hat_D_epsilon[i_p, i_t, i_t ] += 0.0
    hat_D_epsilon[i_p, i_p, i_p ] += 0.0
    
    hat_D_epsilon[i_p, i_r, i_t ] += 0.0
    hat_D_epsilon[i_p, i_t, i_r ] = hat_D_epsilon[i_p, i_r, i_t] 
    
    hat_D_epsilon[i_p, i_r, i_p ] += ( sintheta * h[i_r][i_r] - sintheta * h[i_p][i_p] ) / scaling[i_p]
    hat_D_epsilon[i_p, i_p, i_r ] = hat_D_epsilon[i_p, i_r, i_p]

    hat_D_epsilon[i_p, i_t, i_p ] += 0.0
    hat_D_epsilon[i_p, i_p, i_t ] = hat_D_epsilon[i_p, i_t, i_p]
            
    return hat_D_epsilon
