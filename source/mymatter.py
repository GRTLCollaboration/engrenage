# mymatter.py
# Calculates the matter rhs and stress energy contributions
# this assumes spherical symmetry

import numpy as np
from source.tensoralgebra import *

# params for matter
scalar_mu = 1.0 # this is an inverse length scale related to the scalar compton wavelength
scalar_lambda = 0.0 # dimensionless in geometric units

# The scalar potential
def V_of_u(u) :
    return 0.5 * scalar_mu * scalar_mu * u * u * (1.0 + 0.5 * scalar_lambda * u * u)

# Derivative of scalar potential
def dVdu(u) :
    return scalar_mu * scalar_mu * u * (1.0 + scalar_lambda * u * u)

def get_matter_rhs(u, v, dudr, d2udr2, r_gamma_UU, em4phi, 
                   dphidr, K, lapse, dlapsedr, r_conformal_chris) :
    
    dudt =  lapse * v
    dvdt =  lapse * K * v + r_gamma_UU[i_r] * em4phi * (2.0 * lapse * dphidr * dudr 
                                                      + lapse * d2udr2
                                                      + dlapsedr * dudr)
    for i in range(0, SPACEDIM): 
        dvdt +=  - em4phi * lapse * r_gamma_UU[i] * r_conformal_chris[i_r][i][i] * dudr
    
    # Add mass term
    dvdt += - lapse * dVdu(u)
    
    return dudt, dvdt

def get_rho(u, dudr, v, r_gamma_UU, em4phi) :

    # The potential V(u) = 1/2 mu^2 u^2
    rho = 0.5 * v*v + 0.5 * em4phi * r_gamma_UU[i_r] * dudr * dudr + V_of_u(u)

    return rho

def get_Si(u, dudr, v) :
    
    N = np.size(u)
    S_i = np.zeros([SPACEDIM, N])
    
    S_i[i_r] = - v * dudr
    
    return S_i

# Get rescaled Sij value (rSij = diag[Srr, S_tt / r^2, S_pp / r^2 sin2theta ])
def get_rescaled_Sij(u, dudr, v, r_gamma_UU, em4phi, r_gamma_LL) :

    N = np.size(u)
    rS_ij = np.zeros([SPACEDIM, SPACEDIM, N])

    # Useful quantity Vt
    Vt = - v*v + em4phi * r_gamma_UU[i_r] * (dudr * dudr)
    for i in range(0, SPACEDIM):    
        rS_ij[i][i] = - (0.5 * Vt  + V_of_u(u)) * r_gamma_LL[i] / em4phi + delta[i][i_r] * dudr * dudr
    
    # The trace of S_ij
    S = 0.0
    for i in range(0, SPACEDIM): 
        S += rS_ij[i][i] * r_gamma_UU[i] * em4phi
        
    return S, rS_ij
