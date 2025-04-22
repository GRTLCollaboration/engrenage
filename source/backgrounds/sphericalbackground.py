import numpy as np

from bssn.tensoralgebra import SPACEDIM

# Spherical symmetry for now
sintheta = 1
sin2theta = 1
costheta = 0
cos2theta = 0

# The indices of the metric: r, theta and phi
i_r, i_t, i_p = 0, 1, 2

class FlatSphericalBackground:
    """Represents the background of flat spherical polar coords, defines the hat metric and rescaling."""

    def __init__(self, r) :
        self.r = r
        
        # All other quantities depend on r
        self.N = np.size(r)
        self.scaling_vector = self.get_scaling_vector()
        self.inverse_scaling_vector = self.get_inverse_scaling_vector()
        self.d1_scaling_vector = self.get_d1_scaling_vector()
        self.d1_inverse_scaling_vector = self.get_d1_inverse_scaling_vector()        
        self.d2_scaling_vector = self.get_d2_scaling_vector()
        self.d2_inverse_scaling_vector = self.get_d2_inverse_scaling_vector()        
        self.scaling_matrix = self.get_scaling_matrix()
        self.inverse_scaling_matrix = self.get_inverse_scaling_matrix()
        self.d1_scaling_matrix = self.get_d1_scaling_matrix()
        self.d2_scaling_matrix = self.get_d2_scaling_matrix()
        self.hat_gamma_LL = self.get_hat_gamma_LL()
        self.hat_christoffel = self.get_hat_christoffel()
        self.d1_hat_christoffel = self.get_d1_hat_christoffel()
        self.det_hat_gamma = self.get_det_hat_gamma()
        self.d1_det_hat_gamma = self.get_d1_det_hat_gamma()
        self.d2_det_hat_gamma = self.get_d2_det_hat_gamma()        

    # The vector of scaling factors s_i
    # This is the one that appears in covariant tensor components, e.g. V_i = s_i v_i
    # where v_i is the rescaled quantity and V_i is the physical one
    def get_scaling_vector(self) :
        
        ones = np.ones_like(self.r)
        
        scaling_vector = np.zeros([self.N, SPACEDIM])
        scaling_vector[:,i_r] = ones
        scaling_vector[:,i_t] = self.r
        scaling_vector[:,i_p] = self.r*sintheta
        
        return scaling_vector

    # The vector of inverse scaling factors s^i
    # This is the one that appears in contravariant tensor components, e.g. V^i = s^i v^i
    # where v^i is the rescaled quantity and V^i is the physical one    
    def get_inverse_scaling_vector(self) :
        
        scaling_vector = self.get_scaling_vector()
        inv_scaling_vector = 1.0 / scaling_vector
        
        return inv_scaling_vector
    
    # This is d (scaling_i) / dx^j
    def get_d1_scaling_vector(self) :
        
        ones = np.ones_like(self.r)
    
        ds_dx = np.zeros([self.N, SPACEDIM, SPACEDIM])
        ds_dx[:,i_t, i_r] += ones
        ds_dx[:,i_p, i_r] += ones * sintheta
        ds_dx[:,i_p, i_t] += self.r * costheta
        
        return ds_dx
    
    def get_d1_inverse_scaling_vector(self) :
    
        ds_dx = np.zeros([self.N, SPACEDIM, SPACEDIM])
        ds_dx[:,i_t, i_r] += - 1.0 / self.r / self.r
        ds_dx[:,i_p, i_r] += - 1.0 / self.r / self.r / sintheta
        ds_dx[:,i_p, i_t] += - 1.0 / self.r * costheta / sintheta / sintheta
        
        return ds_dx
    
    def get_d2_inverse_scaling_vector(self) :
    
        d2s_dxdy = np.zeros([self.N, SPACEDIM, SPACEDIM, SPACEDIM])
        d2s_dxdy[:,i_t, i_r, i_r] += 2.0 / (self.r ** 3.0) 
        d2s_dxdy[:,i_p, i_r, i_r] += 2.0 / (self.r ** 3.0) / sintheta
        d2s_dxdy[:,i_p, i_r, i_t] += 1.0 / (self.r ** 2.0 * sintheta ** 2.0) * costheta
        d2s_dxdy[:,i_p, i_t, i_r] += 1.0 / (self.r ** 2.0 * sintheta ** 2.0) * costheta
        d2s_dxdy[:,i_p, i_t, i_t] += 1.0 / (self.r * sintheta ** 3.0) * (costheta ** 2.0 + 1.0)
        
        return d2s_dxdy

    # This is d2 (scaling_i) / dx^j dx^k
    def get_d2_scaling_vector(self) :
        
        ones = np.ones_like(self.r)
    
        d2s_dxdy = np.zeros([self.N, SPACEDIM, SPACEDIM, SPACEDIM])
        d2s_dxdy[:,i_p, i_r, i_t] += ones * costheta
        d2s_dxdy[:,i_p, i_t, i_r] += ones * costheta
        d2s_dxdy[:,i_p, i_t, i_t] += - self.r * sintheta
        
        return d2s_dxdy

    # This is scaling_ij = s_i s_j    
    def get_scaling_matrix(self) :
    
        scaling_vector = self.get_scaling_vector()
        scaling_matrix = np.zeros([self.N, SPACEDIM, SPACEDIM])
        
        for i in np.arange(SPACEDIM) :
            for j in np.arange(SPACEDIM) :
                scaling_matrix[:,i,j] = scaling_vector[:,i] * scaling_vector[:,j]
        
        return scaling_matrix
    
    def get_inverse_scaling_matrix(self) :
        
        scaling_matrix = self.get_scaling_matrix()
        inv_scaling_matrix = 1.0 / scaling_matrix
        
        return inv_scaling_matrix
    
    # This is d (scaling_ij) / dx^k
    def get_d1_scaling_matrix(self) :
    
        ds_dx = self.get_d1_scaling_vector()
        scaling_vector = self.get_scaling_vector()
        dm_dx = np.zeros([self.N, SPACEDIM, SPACEDIM, SPACEDIM])
        
        for i in np.arange(SPACEDIM) :
            for j in np.arange(SPACEDIM) :
                for k in np.arange(SPACEDIM) :
                    dm_dx[:,i,j,k] = scaling_vector[:,i] * ds_dx[:,j,k] + scaling_vector[:,j] * ds_dx[:,i,k]
        
        return dm_dx
    
    # This is d (scaling_ij) / dx^k
    def get_d2_scaling_matrix(self) :
    
        ds_dx = self.get_d1_scaling_vector()
        d2s_dxdy = self.get_d2_scaling_vector()
    
        d2m_dxdy = np.zeros([self.N, SPACEDIM, SPACEDIM, SPACEDIM, SPACEDIM])
        
        for i in np.arange(SPACEDIM) :
            for j in np.arange(SPACEDIM) :
                for k in np.arange(SPACEDIM) :
                    for l in np.arange(SPACEDIM) :
                        d2m_dxdy[:,i,j,k,l] = (ds_dx[:,i,l] * ds_dx[:,j,k] + self.scaling_vector[:,i] * d2s_dxdy[:,j,k,l] +
                                               ds_dx[:,j,l] * ds_dx[:,i,k] + self.scaling_vector[:,j] * d2s_dxdy[:,i,k,l] )
        
        return d2m_dxdy
    
    def get_hat_gamma_LL(self) :
        
        ones = np.ones_like(self.r)
        
        hat_gamma_LL = np.zeros([self.N, SPACEDIM, SPACEDIM])
        hat_gamma_LL[:,i_r,i_r] = ones
        hat_gamma_LL[:,i_t,i_t] = self.r * self.r
        hat_gamma_LL[:,i_p,i_p] = self.r * self.r * sin2theta
        
        return hat_gamma_LL
    
    # christoffel symbols for the hat metric
    # See eqn (18) in Baumgarte https://arxiv.org/abs/1211.6632
    def get_hat_christoffel(self) :
        
        hat_chris = np.zeros([self.N, SPACEDIM, SPACEDIM, SPACEDIM])
        one_over_r = 1.0 / self.r
        
        # non zero r comps \Gamma^r_ab
        hat_chris[:,i_r,i_t,i_t] = - self.r
        hat_chris[:,i_r,i_p,i_p] = - self.r * sin2theta
        
        # non zero theta comps \Gamma^theta_ab
        hat_chris[:,i_t,i_p,i_p] = - sintheta * costheta 
        hat_chris[:,i_t,i_r,i_t] = one_over_r
        hat_chris[:,i_t,i_t,i_r] = one_over_r
    
        # non zero theta comps \Gamma^phi_ab
        hat_chris[:,i_p,i_p,i_r] = one_over_r
        hat_chris[:,i_p,i_r,i_p] = one_over_r
        hat_chris[:,i_p,i_t,i_p] = costheta / sintheta
        hat_chris[:,i_p,i_p,i_t] = costheta / sintheta
        
        return hat_chris
    
    # d \hat Gamma^i_jk / dx^l
    def get_d1_hat_christoffel(self) :
    
        ones = np.ones_like(self.r)
        one_over_r = 1.0 / self.r
        one_over_r2 = one_over_r * one_over_r
        
        d1_hat_chris_dx = np.zeros([self.N, SPACEDIM, SPACEDIM, SPACEDIM, SPACEDIM])
        # non zero r comps \Gamma^r_ab
        d1_hat_chris_dx[:,i_r,i_t,i_t,i_r] = - ones
        d1_hat_chris_dx[:,i_r,i_p,i_p,i_r] = - ones * sin2theta
        d1_hat_chris_dx[:,i_r,i_p,i_p,i_t] = - 2.0 * self.r * sintheta * costheta
        
        # non zero theta comps \Gamma^theta_ab
        d1_hat_chris_dx[:,i_t,i_p,i_p,i_t] = sin2theta - cos2theta
        d1_hat_chris_dx[:,i_t,i_r,i_t,i_r] = - one_over_r2
        d1_hat_chris_dx[:,i_t,i_t,i_r,i_r] = - one_over_r2
    
        # non zero theta comps \Gamma^phi_ab
        d1_hat_chris_dx[:,i_p,i_p,i_r,i_r] = - one_over_r2
        d1_hat_chris_dx[:,i_p,i_r,i_p,i_r] = - one_over_r2
        d1_hat_chris_dx[:,i_p,i_t,i_p,i_t] = - 1.0 / sin2theta
        d1_hat_chris_dx[:,i_p,i_p,i_t,i_t] = - 1.0 / sin2theta
            
        return d1_hat_chris_dx
        
    # determinant of \hat gamma
    def get_det_hat_gamma(self) :
        
        hat_gamma_LL = self.get_hat_gamma_LL()
        det_hat_gamma = np.linalg.det(hat_gamma_LL)        
        
        return det_hat_gamma
    
    # d1 \hat gamma / dx^i
    def get_d1_det_hat_gamma(self) :
    
        d1_det_hat_gamma_dx = np.zeros([self.N, SPACEDIM])  
        d1_det_hat_gamma_dx[:,i_r] = 4.0 * (self.r ** 3.0) * sin2theta 
        d1_det_hat_gamma_dx[:,i_t] = 2.0 * (self.r ** 4.0) * sintheta * costheta
        
        return d1_det_hat_gamma_dx 
    
    # d2 \hat gamma / dx^i dx^j
    def get_d2_det_hat_gamma(self) :
    
        d2_det_hat_gamma_dxdy = np.zeros([self.N, SPACEDIM, SPACEDIM])        
        d2_det_hat_gamma_dxdy[:,i_r,i_r] = 12.0 * (self.r ** 2.0) * sin2theta 
        d2_det_hat_gamma_dxdy[:,i_r,i_t] =  8.0 * (self.r ** 3.0) * sintheta * costheta 
        d2_det_hat_gamma_dxdy[:,i_t,i_r] =  8.0 * (self.r ** 3.0) * sintheta * costheta 
        d2_det_hat_gamma_dxdy[:,i_t,i_t] =  2.0 * (self.r ** 4.0) * (cos2theta - sin2theta)
        
        return d2_det_hat_gamma_dxdy
    
    # end of FlatSphericalBackground class