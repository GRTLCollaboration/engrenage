"""
Set the initial conditions for all the variables for an isotropic Schwarzschild BH.

See further details in https://github.com/GRChombo/engrenage/wiki/Running-the-black-hole-example.
"""

import numpy as np
import pandas as pd
from core.grid import *
from bssn.bssnstatevariables import *
from bssn.tensoralgebra import *
from backgrounds.sphericalbackground import *
from matter.scalarmatter import *

def get_initial_state(grid: Grid, background) :
    
    assert grid.NUM_VARS == 14, "NUM_VARS not correct for bssn + scalar field"
    
    # For readability
    
    r = grid.r
    N = grid.num_points
    
            
      
    
   
   
  
    initial_state = np.zeros((grid.NUM_VARS, N))
    (
        phi,
        hrr,
        htt,
        hpp,
        K,
        arr,
        att,
        app,
        lambdar,
        shiftr,
        br,
        lapse,
        u, 
        v
    ) = initial_state


       
    #grid.r[0]=grid.r[1]
    # Set BH length scale
    GM = 1.0

      # Leggi il CSV e ordina per P
    #df = pd.read_csv("../source/initialdata/collapse_id.csv").sort_values("r")
    df = pd.read_csv("../source/initialdata/dispersion_id.csv").sort_values("r")

    # Funzioni per ricavare alpha_raw, alpha_norm e psi
    #def get_alpha_raw(r):
        #return np.interp(r, df["r"], df["alpha_raw(r)"])

    def get_alpha_norm(a_r):
        return np.interp(a_r, df["r"], df["alpha_norm(r)"])

    def get_psi(a_r):
        return np.interp(a_r, df["r"], df["psi(r)"])
                     
    def get_u(a_r):
        return np.interp(a_r, df["r"], df["u(r)"])



    
    # set non zero metric values
    #grr = (1+ 1/(2.*r))**4 
    grr = np.exp(4.0*get_psi(r))
    gtt_over_r2 = grr
    gpp_over_r2sintheta = gtt_over_r2
    phys_gamma_over_r4sin2theta = grr * gtt_over_r2 * gpp_over_r2sintheta
    
    #phi[:] = 0.25*np.log(grr)
    phi[:] = get_psi(r)
    # Note sign error in Baumgarte eqn (2), conformal factor
    u[:]= get_u(r)
    v[:]= 0.0
    
    phi[:] = np.clip(phi, None, 10.0)
    em4phi = np.exp(-4.0*phi)
    hrr[:] = em4phi * grr - 1.0
    htt[:] = em4phi * gtt_over_r2 - 1.0
    hpp[:] = em4phi * gpp_over_r2sintheta - 1.0    
    lapse.fill(1.0)
    #lapse[:] = get_alpha_norm(r) 
    #lapse[:] = np.clip(get_alpha_norm(r), 1e-6, 10.0)
   
    # overwrite inner cells using parity under r -> - r
    grid.fill_inner_boundary(initial_state)
    
    # Set up matrices
    zeros = np.zeros_like(hrr)
    h_LL = np.array([[hrr, zeros, zeros],[zeros, htt, zeros],[zeros, zeros, hpp]])
    h_LL = np.moveaxis(h_LL, -1, 0) 
    first_derivative_indices = [idx_hrr, idx_htt, idx_hpp]
    dstate_dr = grid.get_first_derivative(initial_state, first_derivative_indices)
    (dhrr_dr, dhtt_dr, dhpp_dr) = dstate_dr[first_derivative_indices]
        
    # This is d h_ij / dx^k = dh_dx[x,i,j,k]
    d1_h_dx = np.zeros([N, SPACEDIM, SPACEDIM, SPACEDIM])
    d1_h_dx[:,i_r,i_r, i_r]  = dhrr_dr
    d1_h_dx[:,i_t,i_t, i_r]  = dhtt_dr
    d1_h_dx[:,i_p,i_p, i_r]  = dhpp_dr
        
    # (unscaled) \bar\gamma_ij and \bar\gamma^ij
    bar_gamma_LL = get_bar_gamma_LL(r, h_LL, background)
    bar_gamma_UU = get_bar_gamma_UU(r, h_LL, background)

    # === Rinormalizzazione di bar_gamma_LL ===
    det_bar = np.linalg.det(bar_gamma_LL)
    det_hat = background.det_hat_gamma

    ratio = det_bar / det_hat
    ratio = np.where(ratio > 1e-14, ratio, 1e-14)

    s = ratio**(-1.0/3.0)

    bar_gamma_LL = s[:, None, None] * bar_gamma_LL
   
    # hrr, htt, hpp derivati dal bar_gamma rinormalizzato
    hrr[:] = bar_gamma_LL[:, i_r, i_r] - 1.0
    htt[:] = bar_gamma_LL[:, i_t, i_t] / (r**2) - 1.0
    hpp[:] = bar_gamma_LL[:, i_p, i_p] / (r**2 * sintheta**2) - 1.0

    bar_gamma_UU = get_bar_gamma_UU(r, bar_gamma_LL, background)

    # aggiorna anche le variabili nello state (hrr, htt, hpp)
    #hrr[:] = bar_gamma_LL[:, i_r, i_r] - background.hat_gamma_LL[:, i_r, i_r]
    #htt[:] = bar_gamma_LL[:, i_t, i_t] / (r**2) - background.hat_gamma_LL[:, i_t, i_t] / (r**2)
    #hpp[:] = bar_gamma_LL[:, i_p, i_p] / (r**2 * sintheta**2) - background.hat_gamma_LL[:, i_p, i_p] / (r**2 * sintheta**2)
         
    # The connections Delta^i, Delta^i_jk and Delta_ijk
    Delta_U, Delta_ULL, Delta_LLL  = get_tensor_connections(r, h_LL, d1_h_dx, background)
    lambdar[:]   = Delta_U[:,i_r]

    # Fill boundary cells for lambdar
    grid.fill_outer_boundary(initial_state, [idx_lambdar])

    # overwrite inner cells using parity under r -> - r
    grid.fill_inner_boundary(initial_state, [idx_lambdar])
            
    return initial_state.reshape(-1)
