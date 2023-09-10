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

# Kronecker delta \delta_ij (i.e. identity matrix)
delta = np.identity(SPACEDIM)

# flat spherical christoffel symbols
# See eqn (18) in Baumgarte https://arxiv.org/abs/1211.6632
def get_flat_spherical_chris(r) :
    
    N = np.size(r)
    spherical_chris = np.zeros([SPACEDIM, SPACEDIM, SPACEDIM, N])
    one_over_r = 1.0 / r
    
    # non zero r comps \Gamma^r_ab
    spherical_chris[i_r][i_t][i_t][:] = - r
    spherical_chris[i_r][i_p][i_p][:] = - r * sin2theta
    
    # non zero theta comps \Gamma^theta_ab
    spherical_chris[i_t][i_p][i_p][:] = - sintheta * costheta 
    spherical_chris[i_t][i_r][i_t][:] = one_over_r
    spherical_chris[i_t][i_t][i_r][:] = one_over_r

    # non zero theta comps \Gamma^phi_ab
    spherical_chris[i_p][i_p][i_r][:] = one_over_r
    spherical_chris[i_p][i_r][i_p][:] = one_over_r
    spherical_chris[i_p][i_t][i_p][:] = costheta / sintheta
    spherical_chris[i_p][i_p][i_t][:] = costheta / sintheta
    
    return spherical_chris

# flat spherical christoffel symbols rescaled for r^2 factors 
# (as if they were tensors) as in Baumgarte https://arxiv.org/abs/1211.6632
def get_rescaled_flat_spherical_chris(r) :

    N = np.size(r)
    spherical_chris = np.zeros([SPACEDIM, SPACEDIM, SPACEDIM, N])
    one_over_r = 1.0 / r
    
    # non zero r comps \Gamma^r_ab
    spherical_chris[i_r][i_t][i_t][:] = - one_over_r
    spherical_chris[i_r][i_p][i_p][:] = - one_over_r
    
    # non zero theta comps \Gamma^theta_ab
    spherical_chris[i_t][i_p][i_p][:] = - costheta * one_over_r 
    spherical_chris[i_t][i_r][i_t][:] = one_over_r
    spherical_chris[i_t][i_t][i_r][:] = one_over_r

    # non zero theta comps \Gamma^phi_ab
    spherical_chris[i_p][i_p][i_r][:] = one_over_r
    spherical_chris[i_p][i_r][i_p][:] = one_over_r
    spherical_chris[i_p][i_t][i_p][:] = costheta / sintheta * one_over_r
    spherical_chris[i_p][i_p][i_t][:] = costheta / sintheta * one_over_r
    
    return spherical_chris

def get_conformal_chris(Delta_ULL, r) :

    return get_flat_spherical_chris(r) + Delta_ULL

def get_rescaled_conformal_chris(rDelta_ULL, r_here) :

    return get_rescaled_flat_spherical_chris(r) + rDelta_ULL
    
# Compute determinant of conformal spatial metric \bar\gamma divided by r^4 sin^2 theta
# Assumes h_metric is diagonal so 3 elements
def get_rescaled_determinant_gamma(h_tensor) :
    
    determinant = (1.0 + h_tensor[i_r][:]) * (1.0 + h_tensor[i_t][:]) * (1.0 + h_tensor[i_p][:])
    
    return determinant

# Computer the conformal spatial metric \bar \gamma_ij given the rescaled perturbation h
# Assumes h_metric is diagonal so 3 elements
def get_metric(r, h_tensor) :
    
    scaling = np.array([np.ones_like(r), r , r*sintheta]) 
    
    gamma_LL = scaling * scaling * (1.0 + h_tensor)
    
    return gamma_LL

# Computer the rescaled spatial metric inv_scaling_ij * \bar \gamma_ij given the rescaled perturbation h
def get_rescaled_metric(h_tensor) :   
    
    r_gamma_LL = 1.0 + h_tensor 
    
    return r_gamma_LL

# Computer inverse of the conformal spatial metric \bar\gamma^ij given the rescaled perturbation h
# Assumes h_metric is diagonal so 3 elements
def get_inverse_metric(r, h_metric):
    
    inv_scaling = np.array([np.ones_like(r), 1.0/r , 1.0/r/sintheta])
    return inv_scaling * inv_scaling / (1. + h_metric)

def get_rescaled_inverse_metric(h_metric):
    
    return  1. / (1. + h_metric)

# Compute the \bar A_ij given the rescaled perturbation a
# assumes a diagonal so 3 elements
def get_A_LL(r, a):

    scaling = np.array([np.ones_like(r), r , r*sintheta])    
    return scaling * scaling * a 

# Compute the \bar A^ij given a_ij and the reduced \bar\gamma^ij
# assumes a and gamma diagonal so 3 elements
def get_A_UU(a, r_gamma_UU, r):

    inv_scaling = np.array([np.ones_like(r), 1.0/r , 1.0/r/sintheta])
    
    a_UU = inv_scaling * inv_scaling * r_gamma_UU * a * r_gamma_UU 
    
    return a_UU

# Compute a^ij given a_ij
# assumes a and gamma diagonal so 3 elements
def get_a_UU(a, r_gamma_UU):
    
    a_UU = r_gamma_UU * a * r_gamma_UU  
    
    return a_UU

# Compute trace of (traceless part of) extrinsic curvature
# assumes a and gamma diagonal so 3 elements
def get_trace_A(a, r_gamma_UU) :

    # Matrix multiply, then matrix trace
    return np.sum(r_gamma_UU * a, axis=0)

# Compute trace of some rank 2 tensor with indices lowered
# assumes both diagonal so 3 elements
def get_trace(T_LL, gamma_UU):

    # Matrix multiply, then matrix trace
    return np.sum(gamma_UU * T_LL, axis=0)

# Compute A_ij A^ij
# assumes a and gamma diagonal so 3 elements
def get_Asquared(a, r_gamma_UU) :
    
    Asquared = np.sum(a * a * r_gamma_UU * r_gamma_UU, axis=0)

    return Asquared

# Compute connection of the spatial metric
# See eqn (23)-(24) in Baumgarte https://arxiv.org/abs/1211.6632
# \Delta^i_{jk} \equiv \bar \Gamma^i_{jk} - \Gamma0^i_{jk}
#               = (1/2) * \bar \gamma^{il} ( hat D_j \bar \gamma_{kl} + hat D_k \bar \gamma_{jl} - hat D_l \bar \gamma_{jk} )
#               = (1/2) * \bar \gamma^{il} ( hat D_j \epsilon_{kl} + hat D_k \epsilon_{jl} - hat D_l \epsilon_{jk} )
# where hat D_i is the covariant derivative associated with the flat metric \eta_{ij}.
def get_connection(r, bar_gamma_UU, bar_gamma_LL, h, dhdr) :

    N = np.size(r)
    Delta_ULL = np.zeros([SPACEDIM, SPACEDIM, SPACEDIM, N])
    Delta_LLL = np.zeros([SPACEDIM, SPACEDIM, SPACEDIM, N])
    hat_D_bar_gamma = get_hat_D_bar_gamma(r, h, dhdr)
    
    for i in range(0, SPACEDIM):
        for j in range(0, SPACEDIM):    
            for k in range(0, SPACEDIM):
                Delta_ULL[i][j][k][:] += 0.5 * bar_gamma_UU[i][:] * (  hat_D_bar_gamma[j][k][i][:]
                                                                  +    hat_D_bar_gamma[k][j][i][:] 
                                                                  -    hat_D_bar_gamma[i][j][k][:])
                    
    Delta_U =  np.zeros([SPACEDIM, N])         
    for i in range(0, SPACEDIM):
        for j in range(0, SPACEDIM):    
            Delta_U[i] += bar_gamma_UU[j][:] * Delta_ULL[i][j][j][:]
            for k in range(0, SPACEDIM):
                for m in range(0, SPACEDIM):
                    Delta_LLL[i][j][k][:] += delta[m][i]*bar_gamma_LL[i][:] * Delta_ULL[m][j][k][:]
    
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
def get_ricci_tensor(r, h, dhdr, d2hdr2, lambdar, dlambardr, 
                     Delta_U, Delta_ULL, Delta_LLL, 
                     bar_gamma_UU, bar_gamma_LL) :

    N = np.size(r)    
    ricci = np.zeros([SPACEDIM, SPACEDIM, N])
    
    # Get \hat D \Lambda^i
    hat_D_Lambda = get_hat_D_Lambda(r, lambdar, dlambardr)
    # Get \bar\gamma^kl \hat D_k \hat D_l \bar\gamma_ij
    hat_D2_bar_gamma = get_hat_D2_bar_gamma(r, h, dhdr, d2hdr2, bar_gamma_UU)
    
    for i in range(0, SPACEDIM):
        for j in range(0, SPACEDIM):
            ricci[i][j][:] += - 0.5 * hat_D2_bar_gamma[i][j][:]
            
            for k in range(0, SPACEDIM):
                ricci[i][j][:] += ( 0.5 * (delta[k][i] * bar_gamma_LL[i] * hat_D_Lambda[j][k] + 
                                           delta[k][j] * bar_gamma_LL[j] * hat_D_Lambda[i][k])
                                  + 0.5 * (Delta_U[k] * Delta_LLL[i][j][k] + 
                                           Delta_U[k] * Delta_LLL[j][i][k]))
                for l in range(0, SPACEDIM):               
                    for m in range(0, SPACEDIM): 
                        ricci[i][j][:] += delta[k][l] * bar_gamma_UU[k] * (  Delta_ULL[m][k][i][:] * 
                                                                             Delta_LLL[j][m][l][:]
                                                                           + Delta_ULL[m][k][j][:] * 
                                                                             Delta_LLL[i][m][l][:]
                                                                           + Delta_ULL[m][i][k][:] * 
                                                                             Delta_LLL[m][j][l][:] )
            
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
def get_hat_D_Lambda(r, lambdar, dlambardr) :
    
    # Make an array for \hat D_i \Lambda^j
    N = np.size(r)    
    hat_D_Lambda = np.zeros([SPACEDIM, SPACEDIM, N])
    
    # Useful quantities
    flat_chris = get_flat_spherical_chris(r)

    hat_D_Lambda[i_r][i_r][:] = dlambardr
    for i in range(0, SPACEDIM):
        for j in range(0, SPACEDIM):
            hat_D_Lambda[i][j][:] += flat_chris[j][i][i_r][:] * lambdar
                
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
def get_hat_D2_bar_gamma(r, h, dhdr, d2hdr2, bar_gamma_UU) :
    
    N = np.size(r)    
    hat_D2_bar_gamma = np.zeros([SPACEDIM, SPACEDIM, N])
    
    # Useful quantities
    hat_D_bar_gamma = get_hat_D_bar_gamma(r, h, dhdr)
    flat_chris = get_flat_spherical_chris(r)
    r2 = r * r
    r2sin2theta = r2 * sin2theta
    
    # explicitly add non zero terms in spherical symmetry
    hat_D2_bar_gamma[i_r][i_r][:] = bar_gamma_UU[i_r] *   d2hdr2[i_r]
    
    hat_D2_bar_gamma[i_t][i_t][:] = bar_gamma_UU[i_r] * ( d2hdr2[i_t] * r2
                                                            + dhdr[i_t] * 2.0 * r )
    hat_D2_bar_gamma[i_p][i_p][:] = bar_gamma_UU[i_r] * ( d2hdr2[i_p] * r2sin2theta 
                                                            + dhdr[i_p] * 2.0 * r * sin2theta )
    
    # now add the christoffel terms
    for i in range(0, SPACEDIM):
        for j in range(0, SPACEDIM):            
            for k in range(0, SPACEDIM):   
                for l in range(0, SPACEDIM):
                    for m in range(0, SPACEDIM): 
                        hat_D2_bar_gamma[i][j][:] += (- bar_gamma_UU[k] * delta[k][l] * 
                                                                         (  hat_D_bar_gamma[m][i][j] * flat_chris[m][l][k]
                                                                          + hat_D_bar_gamma[l][m][j] * flat_chris[m][i][k]
                                                                          + hat_D_bar_gamma[l][i][m] * flat_chris[m][j][k] ))
                                                                        
                
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
def get_hat_D_bar_gamma(r, h, dhdr) :

    N = np.size(r)
    hat_D_epsilon = np.zeros([SPACEDIM, SPACEDIM, SPACEDIM, N])
    
    # assume spherical symmetry
    dhdtheta = np.zeros_like(dhdr)
    dhdphi = np.zeros_like(dhdr)
    
    # some useful quantities
    r2 = r * r
    scaling = np.array([np.ones_like(r), r , r*sintheta])
    
    # Fill derivatives \hat D_k epsilon_ij
    for i in range(0, SPACEDIM):
        hat_D_epsilon[i_r, i, i][:]   = dhdr[i][:]     * scaling[i] * scaling[i]
        hat_D_epsilon[i_t, i, i][:]   = dhdtheta[i][:] * scaling[i] * scaling[i]
        hat_D_epsilon[i_p, i, i][:]   = dhdphi[i][:]   * scaling[i] * scaling[i]
            
    # Add additional terms from christoffels etc
    
    # d/dtheta
    hat_D_epsilon[i_t, i_r, i_r ][:] +=   0.0
    hat_D_epsilon[i_t, i_t, i_t ][:] +=   0.0
    hat_D_epsilon[i_t, i_p, i_p ][:] +=   0.0
    
    hat_D_epsilon[i_t, i_r, i_t ][:] += scaling[i_r] * scaling[i_t] * (  h[i_r] - h[i_t] )
    hat_D_epsilon[i_t, i_t, i_r ][:] = hat_D_epsilon[i_t, i_r, i_t][:] 
    
    hat_D_epsilon[i_t, i_r, i_p ][:] += 0.0
    hat_D_epsilon[i_t, i_p, i_r ][:] = hat_D_epsilon[i_t, i_r, i_p][:]

    hat_D_epsilon[i_t, i_t, i_p ][:] += 0.0
    hat_D_epsilon[i_t, i_p, i_t ][:] = hat_D_epsilon[i_t, i_t, i_p][:]
    
    # d/dphi
    hat_D_epsilon[i_p, i_r, i_r ][:] +=   0.0
    hat_D_epsilon[i_p, i_t, i_t ][:] +=   0.0
    hat_D_epsilon[i_p, i_p, i_p ][:] +=   0.0
    
    hat_D_epsilon[i_p, i_r, i_t ][:] += 0.0
    hat_D_epsilon[i_p, i_t, i_r ][:] = hat_D_epsilon[i_p, i_r, i_t][:]
    
    hat_D_epsilon[i_p, i_r, i_p ][:] += scaling[i_r] * scaling[i_p] * (sintheta * h[i_r]
                                                                     - sintheta * h[i_p] )
    hat_D_epsilon[i_p, i_p, i_r ][:] = hat_D_epsilon[i_p, i_r, i_p][:]

    hat_D_epsilon[i_p, i_t, i_p ][:] += scaling[i_t] * scaling[i_p] * (costheta * h[i_t]
                                                                     - costheta * h[i_p] )
    hat_D_epsilon[i_p, i_p, i_t ][:] = hat_D_epsilon[i_p, i_t, i_p][:]
            
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
