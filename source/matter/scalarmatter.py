import numpy as np

from core.grid import *
from bssn.bssnstatevariables import *
from bssn.bssnvars import *
from bssn.tensoralgebra import *

class ScalarMatter :
    """Represents the matter that sources the Einstein equation."""

    def __init__(self, a_scalar_mu=1.0) :
        self.scalar_mu = a_scalar_mu # this is an inverse length scale related to the scalar compton wavelength
        
        # Details for the matter state variables
        self.NUM_MATTER_VARS = 2
        self.VARIABLE_NAMES = ["u", "v"]
        self.PARITY = np.array([1, 1])
        self.ASYMP_POWER = np.array([0, 0])
        self.ASYMP_OFFSET = np.array([0, 0])
        self.idx_u = NUM_BSSN_VARS
        self.idx_v = NUM_BSSN_VARS + 1
        self.indices = np.array([self.idx_u, self.idx_v])
        self.matter_vars_set = False
        self.u = []
        self.v = []
        self.d1_u = []
        self.d2_u = []
        self.advec_u = []
        self.advec_v = []
        
    # The scalar potential
    def V_of_u(self, u) :
        return 0.5 * self.scalar_mu * self.scalar_mu * u * u

    # Derivative of scalar potential
    def dVdu(self, u) :
        return self.scalar_mu * self.scalar_mu * u
    
    def get_emtensor(self, r, bssn_vars, background) :
    
        assert self.matter_vars_set, 'Matter vars not set'
        
        N = np.size(r) 
        scalar_emtensor = EMTensor(N)
        
        em4phi = np.exp(-4.0 * bssn_vars.phi)
        bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)
        # The potential V(u) = 1/2 mu^2 u^2
        scalar_emtensor.rho = (  0.5 * self.v * self.v
                               + 0.5 * em4phi * np.einsum('xij,xi,xj->x', bar_gamma_UU, self.d1_u, self.d1_u)
                               + self.V_of_u(self.u) )
        
        scalar_emtensor.Si = - self.v[:,np.newaxis] * self.d1_u
        
        # Useful quantity Vt
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
        Vt = - self.v * self.v + em4phi * np.einsum('xij,xi,xj->x', bar_gamma_UU, self.d1_u, self.d1_u)
        
        # Need to get the scalar factor in the right array dimension
        scalar_factor = - ((0.5 * Vt  + self.V_of_u(self.u)) / em4phi)
        scalar_emtensor.Sij = (scalar_factor[:,np.newaxis,np.newaxis] * bar_gamma_LL
                                   + np.einsum('xi,xj->xij', self.d1_u, self.d1_u))
        
        # The trace of S_ij
        scalar_emtensor.S = em4phi * np.einsum('xjk,xjk->x', bar_gamma_UU, scalar_emtensor.Sij)        
            
        return scalar_emtensor

    def get_matter_rhs(self, r, bssn_vars, bssn_d1, background) :

        assert self.matter_vars_set, 'Matter vars not set'        
        
        # The connections Delta^i, Delta^i_jk and Delta_ijk
        Delta_U, Delta_ULL, Delta_LLL  = get_tensor_connections(r, bssn_vars.h_LL, bssn_d1.h_LL, background)
        
        # \bar \Gamma^i_jk
        bar_chris = get_bar_christoffel(r, Delta_ULL, background) 
        
        em4phi = np.exp(-4.0*bssn_vars.phi)    
        bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)
        
        dudt =  bssn_vars.lapse * self.v
        dvdt =  (bssn_vars.lapse * bssn_vars.K * self.v 
                 + 2.0 * bssn_vars.lapse * em4phi * np.einsum('xij,xi,xj->x', bar_gamma_UU, bssn_d1.phi, self.d1_u)
                 +       bssn_vars.lapse * em4phi * np.einsum('xij,xij->x', bar_gamma_UU, self.d2_u)
                 +                         em4phi * np.einsum('xij,xi,xj->x', bar_gamma_UU, bssn_d1.lapse, self.d1_u)
                 -       bssn_vars.lapse * em4phi * np.einsum('xij,xkij,xk->x', bar_gamma_UU, bar_chris, self.d1_u))
        
        # Add mass term
        dvdt += - bssn_vars.lapse * self.dVdu(self.u)
        
        # Now advection
        dudt   += np.einsum('xj,xj->x', background.inverse_scaling_vector * bssn_vars.shift_U,   self.advec_u)
        dvdt   += np.einsum('xj,xj->x', background.inverse_scaling_vector * bssn_vars.shift_U,   self.advec_v)
        
        return dudt, dvdt
    
    # Set the matter vars and their derivs from the full state vector
    def set_matter_vars(self, state_vector, bssn_vars : BSSNVars, grid : Grid) :
         
        (self.u, self.v) = state_vector[self.idx_u], state_vector[self.idx_v]
        
        # get the derivatives of u needed for the evolution
        # need to get rid of the dependence on the background here somewhow...
        self.d1_u = np.zeros([grid.N, SPACEDIM])
        self.d2_u = np.zeros([grid.N, SPACEDIM, SPACEDIM]) 
        d1_state = grid.get_first_derivative(state_vector, [self.idx_u] )
        self.d1_u[:,i_x1] = d1_state[self.idx_u]
        d2_state = grid.get_second_derivative(state_vector, [self.idx_u])
        self.d2_u[:,i_x1,i_x1] = d2_state[self.idx_u]
        
        # Advective derivs
        advec_state = grid.get_advection(state_vector, bssn_vars.shift_U[:,i_x1] >= 0, self.indices)
        self.advec_u = np.zeros([grid.N, SPACEDIM])
        self.advec_v = np.zeros([grid.N, SPACEDIM])
        self.advec_u[:,i_x1], self.advec_v[:,i_x1] = advec_state[self.indices]
        
        self.matter_vars_set = True
        