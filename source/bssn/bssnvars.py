import numpy as np

from core.statevector import *
from backgrounds.sphericalbackground import *

# These classes just allow us to hold all the metric and derivative quantities in a nice packaged form
# and map the non trivial components into full vector and tensor forms

class BSSNVars:
    """
    Holder for the BSSN quantities in tensor form
    """
    def __init__(self, N):

        self.N = N
        self.h_LL = np.zeros([N, SPACEDIM, SPACEDIM])
        self.a_LL = np.zeros([N, SPACEDIM, SPACEDIM])
        self.lambda_U = np.zeros([N, SPACEDIM])
        self.shift_U = np.zeros([N, SPACEDIM])
        self.b_U = np.zeros([N, SPACEDIM])
        self.lapse = np.zeros([N])
        self.phi = np.zeros([N])
        self.K = np.zeros([N])
        
    def set_bssn_vars(self, state):
        """
        Populate the bssn vars from the full state
        """

        # Unpack variables from current_state - see statevector.py
        # Assumes BSSN vars are first, which they should be
        (phi, hrr, htt, hpp, K, arr, att, app, lambdar, shiftr, br, lapse) = state[0 : NUM_BSSN_VARS]

        self.h_LL[:,i_r,i_r] = hrr
        self.h_LL[:,i_t,i_t] = htt
        self.h_LL[:,i_p,i_p] = hpp
        
        self.a_LL[:,i_r,i_r] = arr
        self.a_LL[:,i_t,i_t] = att
        self.a_LL[:,i_p,i_p] = app
        
        self.lambda_U[:,i_r] = lambdar            
        self.shift_U[:,i_r] = shiftr
        self.b_U[:,i_r] = br

        self.lapse[:] = lapse
        self.phi[:] = phi     
        self.K[:] = K

    def set_bssn_state_vars(self):
        """
        Populate the state vector from the bssn vars values
        """
        
        bssn_state = np.zeros(self.N * NUM_BSSN_VARS)
        bssn_state = bssn_state.reshape(NUM_BSSN_VARS, -1)
        
        # assign parts of the state vector to the different vars
        (
            state_phi,
            state_hrr,
            state_htt,
            state_hpp,
            state_K,
            state_arr,
            state_att,
            state_app,
            state_lambdar,
            state_shiftr,
            state_br,
            state_lapse
        ) = bssn_state
        
        state_phi[:] = self.phi
        state_hrr[:] = self.h_LL[:,i_r,i_r]
        state_htt[:] = self.h_LL[:,i_t,i_t]
        state_hpp[:] = self.h_LL[:,i_p,i_p]
        state_K[:] = self.K
        state_arr[:] = self.a_LL[:,i_r,i_r]
        state_att[:] = self.a_LL[:,i_t,i_t]
        state_app[:] = self.a_LL[:,i_p,i_p]
        state_lambdar[:] = self.lambda_U[:,i_r]
        state_shiftr[:] = self.shift_U[:,i_r]
        state_br[:] = self.b_U[:,i_r]
        state_lapse[:] = self.lapse
        
        return bssn_state
        
        
class BSSNFirstDerivs:
    """
    Holder for the BSSN d1 quantities in tensor form, last indices are the derivative ones by convention
    """
    def __init__(self, N):
        
        self.first_derivative_indices = [idx_phi, idx_hrr, idx_htt, idx_hpp, idx_K, idx_arr, idx_att, idx_app, 
                                         idx_lambdar, idx_shiftr, idx_lapse]

        self.h_LL = np.zeros([N, SPACEDIM, SPACEDIM, SPACEDIM])
        self.a_LL = np.zeros([N, SPACEDIM, SPACEDIM, SPACEDIM])
        self.shift_U = np.zeros([N, SPACEDIM, SPACEDIM])
        self.lambda_U = np.zeros([N, SPACEDIM, SPACEDIM])
        self.phi = np.zeros([N, SPACEDIM])
        self.K = np.zeros([N, SPACEDIM])
        self.lapse = np.zeros([N, SPACEDIM])

        
    def set_bssn_first_derivs(self, dstate_dr):
        """
        Populate the bssn vars from the variable derivs
        """

        (   dphi_dr,
            dhrr_dr,
            dhtt_dr,
            dhpp_dr,
            dK_dr,
            darr_dr,
            datt_dr,
            dapp_dr,
            dlambdar_dr,
            dshiftr_dr,
            dlapse_dr,
        ) = dstate_dr[self.first_derivative_indices]        
        
        self.h_LL[:,i_r,i_r,i_r] = dhrr_dr
        self.h_LL[:,i_t,i_t,i_r] = dhtt_dr
        self.h_LL[:,i_p,i_p,i_r] = dhpp_dr

        self.a_LL[:,i_r,i_r,i_r] = darr_dr
        self.a_LL[:,i_t,i_t,i_r] = datt_dr
        self.a_LL[:,i_p,i_p,i_r] = dapp_dr        

        
        self.shift_U[:,i_r,i_r] = dshiftr_dr 
        self.lambda_U[:,i_r,i_r] = dlambdar_dr
        
        self.phi[:,i_r] = dphi_dr
        self.K[:,i_r] = dK_dr
        self.lapse[:,i_r] = dlapse_dr


class BSSNSecondDerivs:
    """
    Holder for the BSSN d2 quantities in tensor form, last indices are the derivative ones by convention
    """
    def __init__(self, N):
        
        self.second_derivative_indices = [idx_phi, idx_hrr, idx_htt, idx_hpp, idx_shiftr, idx_lapse]

        self.h_LL = np.zeros([N, SPACEDIM, SPACEDIM, SPACEDIM, SPACEDIM])
        self.shift_U = np.zeros([N, SPACEDIM, SPACEDIM, SPACEDIM])
        self.phi = np.zeros([N, SPACEDIM, SPACEDIM])
        self.lapse = np.zeros([N, SPACEDIM, SPACEDIM])
        
    def set_bssn_second_derivs(self, d2state_dr2):
        """
        Populate the bssn vars from the variable derivs
        """
        
        (
            d2phi_dr2,
            d2hrr_dr2,
            d2htt_dr2,
            d2hpp_dr2,
            d2shiftr_dr2,
            d2lapse_dr2,
        ) = d2state_dr2[self.second_derivative_indices]

        self.h_LL[:,i_r,i_r,i_r,i_r]  = d2hrr_dr2
        self.h_LL[:,i_t,i_t,i_r,i_r]  = d2htt_dr2
        self.h_LL[:,i_p,i_p,i_r,i_r]  = d2hpp_dr2  
        
        self.shift_U[:,i_r,i_r,i_r] = d2shiftr_dr2
        
        self.phi[:,i_r,i_r] = d2phi_dr2
        self.lapse[:,i_r,i_r] = d2lapse_dr2

class BSSNAdvecDerivs:
    """
    Holder for the BSSN d1 advec quantities in tensor form, last indices are the derivative ones by convention
    """
    def __init__(self, N):
        
        self.advec_indices = [idx_phi, idx_hrr, idx_htt, idx_hpp, 
                              idx_arr, idx_att, idx_app, idx_K, idx_lambdar, idx_lapse]

        self.h_LL = np.zeros([N, SPACEDIM, SPACEDIM, SPACEDIM])
        self.a_LL = np.zeros([N, SPACEDIM, SPACEDIM, SPACEDIM])
        self.lambda_U = np.zeros([N, SPACEDIM, SPACEDIM])
        self.shift_U = np.zeros([N, SPACEDIM, SPACEDIM])
        self.phi = np.zeros([N, SPACEDIM])
        self.K = np.zeros([N, SPACEDIM])
        self.lapse = np.zeros([N, SPACEDIM])
        
        #temp
        self.u = np.zeros([N, SPACEDIM])
        
    def set_bssn_advec_derivs(self, dstate_dr_advec):
        """
        Populate the bssn vars from the variable derivs
        """

        (
            dphi_dr_advec,
            dhrr_dr_advec,
            dhtt_dr_advec,
            dhpp_dr_advec,
            darr_dr_advec,
            datt_dr_advec,
            dapp_dr_advec,
            dK_dr_advec,
            dlambdar_dr_advec,
            dlapse_dr_advec,
        ) = dstate_dr_advec[self.advec_indices]      
        
        self.h_LL[:,i_r,i_r,i_r] = dhrr_dr_advec
        self.h_LL[:,i_t,i_t,i_r] = dhtt_dr_advec
        self.h_LL[:,i_p,i_p,i_r] = dhpp_dr_advec
        
        self.a_LL[:,i_r,i_r,i_r] = darr_dr_advec
        self.a_LL[:,i_t,i_t,i_r] = datt_dr_advec
        self.a_LL[:,i_p,i_p,i_r] = dapp_dr_advec        
        
        self.lambda_U[:,i_r,i_r] = dlambdar_dr_advec
        
        self.phi[:,i_r] = dphi_dr_advec
        self.K[:,i_r] = dK_dr_advec
        self.lapse[:,i_r] = dlapse_dr_advec