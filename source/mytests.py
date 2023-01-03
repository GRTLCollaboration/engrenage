# mytests.py

# set the initial conditions for all the variables

from source.uservariables import *
from source.tensoralgebra import *
from source.fourthorderderivatives import *
import numpy as np
from scipy.interpolate import interp1d

# This routine gives us something where phi = 0 initially but bar_R and lambda are non trivial
def get_test_vars_values_1(R, N_r) :
    
    # Set up grid values
    dx = R/N_r
    N = N_r + num_ghosts * 2 
    r = np.linspace(-(num_ghosts-0.5)*dx, R+(num_ghosts-0.5)*dx, N)
    oneoverdx  = 1.0 / dx
    oneoverdxsquared = oneoverdx * oneoverdx

    test_vars_values = np.zeros(NUM_VARS * N)
    
    # Some test data with known values of bar_R etc
    for ix in range(num_ghosts, N-num_ghosts) :

        # position on the grid
        r_i = r[ix]

        # scalar field values
        test_vars_values[ix + idx_u * N] = 0.0 # start at a moment where field is zero
        test_vars_values[ix + idx_v * N] = 0.0
 
        # non zero metric variables (note h_rr etc are rescaled difference from flat space so zero
        # and conformal factor is zero for flat space)
        test_vars_values[ix + idx_lapse * N] = 1.0
        # note that we choose that the determinant \bar{gamma} = \hat{gamma} initially
        grr = 1.0 + r_i * r_i * np.exp(-r_i)
        gtt_over_r2 = grr**(-0.5)
        # The following is required for spherical symmetry
        gpp_over_r2sintheta = gtt_over_r2
        phys_gamma_over_r4sin2theta = grr * gtt_over_r2 * gpp_over_r2sintheta
        phi_here = 1.0/12.0 * np.log(phys_gamma_over_r4sin2theta)
        test_vars_values[ix + idx_phi * N]   = phi_here
        em4phi = np.exp(-4.0*phi_here)
        test_vars_values[ix + idx_hrr * N]   = em4phi * grr - 1.0
        test_vars_values[ix + idx_htt * N]   = em4phi * gtt_over_r2 - 1.0
        test_vars_values[ix + idx_hpp * N]   = em4phi * gpp_over_r2sintheta - 1.0
        
    # overwrite outer boundaries with extrapolation (zeroth order)
    for ivar in range(0, NUM_VARS) :
        boundary_cells = np.array([(ivar + 1)*N-3, (ivar + 1)*N-2, (ivar + 1)*N-1])
        for count, ix in enumerate(boundary_cells) :
            offset = -1 - count
            test_vars_values[ix]    = test_vars_values[ix + offset]

    # overwrite inner cells using parity under r -> - r
    for ivar in range(0, NUM_VARS) :
        boundary_cells = np.array([(ivar)*N, (ivar)*N+1, (ivar)*N+2])
        var_parity = parity[ivar]
        for count, ix in enumerate(boundary_cells) :
            offset = 5 - 2*count
            test_vars_values[ix] = test_vars_values[ix + offset] * var_parity           

    # needed for lambdar
    hrr    = test_vars_values[idx_hrr * N : (idx_hrr + 1) * N]
    htt    = test_vars_values[idx_htt * N : (idx_htt + 1) * N]
    hpp    = test_vars_values[idx_hpp * N : (idx_hpp + 1) * N]
    dhrrdx     = get_dfdx(hrr, oneoverdx)
    dhttdx     = get_dfdx(htt, oneoverdx)
    dhppdx     = get_dfdx(hpp, oneoverdx)
    
    # assign lambdar values
    for ix in range(num_ghosts, N-num_ghosts) :
        
        # position on the grid
        r_here = r[ix]
        
        # Assign BSSN vars to local tensors
        h = np.zeros_like(rank_2_spatial_tensor)
        h[i_r][i_r] = hrr[ix]
        h[i_t][i_t] = htt[ix]
        h[i_p][i_p] = hpp[ix]
        
        dhdr = np.zeros_like(rank_2_spatial_tensor)
        dhdr[i_r][i_r] = dhrrdx[ix]
        dhdr[i_t][i_t] = dhttdx[ix]
        dhdr[i_p][i_p] = dhppdx[ix]
        
        # (unscaled) \bar\gamma_ij and \bar\gamma^ij
        bar_gamma_LL = get_metric(r_here, h)
        bar_gamma_UU = get_inverse_metric(r_here, h)
        
        # The connections Delta^i, Delta^i_jk and Delta_ijk
        Delta_U, Delta_ULL, Delta_LLL  = get_connection(r_here, bar_gamma_UU, bar_gamma_LL, h, dhdr)
        test_vars_values[ix + idx_lambdar * N]   = Delta_U[i_r]

    # Fill boundary cells for lambdar
    boundary_cells = np.array([(idx_lambdar + 1)*N-3, (idx_lambdar + 1)*N-2, (idx_lambdar + 1)*N-1])
    for count, ix in enumerate(boundary_cells) :
        offset = -1 - count
        test_vars_values[ix]    = test_vars_values[ix + offset]
        
    boundary_cells = np.array([(idx_lambdar)*N, (idx_lambdar)*N+1, (idx_lambdar)*N+2])
    for count, ix in enumerate(boundary_cells) :
        offset = 5 - 2*count
        test_vars_values[ix] = test_vars_values[ix + offset] * parity[idx_lambdar]
        
    return r, test_vars_values

# This routine gives us something where bar_R is trivial but phi is non trivial
def get_test_vars_values_2(R, N_r) :
    
    # Set up grid values
    dx = R/N_r
    N = N_r + num_ghosts * 2 
    r = np.linspace(-(num_ghosts-0.5)*dx, R+(num_ghosts-0.5)*dx, N)
    oneoverdx  = 1.0 / dx
    oneoverdxsquared = oneoverdx * oneoverdx

    test_vars_values = np.zeros(NUM_VARS * N)
    
    # Some test data with known values of bar_R etc
    for ix in range(num_ghosts, N-num_ghosts) :

        # position on the grid
        r_i = r[ix]

        # scalar field values
        test_vars_values[ix + idx_u * N] = 0.0 # start at a moment where field is zero
        test_vars_values[ix + idx_v * N] = 0.0
 
        # non zero metric variables (note h_rr etc are rescaled difference from flat space so zero
        # and conformal factor is zero for flat space)
        test_vars_values[ix + idx_lapse * N] = 1.0
        # note that we choose that the determinant \bar{gamma} = \hat{gamma} initially
        f = 1.0 + r_i * r_i * np.exp(-r_i)
        grr = f
        gtt_over_r2 = f
        # The following is required for spherical symmetry
        gpp_over_r2sintheta = gtt_over_r2
        phys_gamma_over_r4sin2theta = grr * gtt_over_r2 * gpp_over_r2sintheta
        phi_here = 1.0/12.0 * np.log(phys_gamma_over_r4sin2theta)
        test_vars_values[ix + idx_phi * N]   = phi_here
        em4phi = np.exp(-4.0*phi_here)
        test_vars_values[ix + idx_hrr * N]   = em4phi * grr - 1.0
        test_vars_values[ix + idx_htt * N]   = em4phi * gtt_over_r2 - 1.0
        test_vars_values[ix + idx_hpp * N]   = em4phi * gpp_over_r2sintheta - 1.0
        
    # overwrite outer boundaries with extrapolation (zeroth order)
    for ivar in range(0, NUM_VARS) :
        boundary_cells = np.array([(ivar + 1)*N-3, (ivar + 1)*N-2, (ivar + 1)*N-1])
        for count, ix in enumerate(boundary_cells) :
            offset = -1 - count
            test_vars_values[ix]    = test_vars_values[ix + offset]

    # overwrite inner cells using parity under r -> - r
    for ivar in range(0, NUM_VARS) :
        boundary_cells = np.array([(ivar)*N, (ivar)*N+1, (ivar)*N+2])
        var_parity = parity[ivar]
        for count, ix in enumerate(boundary_cells) :
            offset = 5 - 2*count
            test_vars_values[ix] = test_vars_values[ix + offset] * var_parity           

    # needed for lambdar
    hrr    = test_vars_values[idx_hrr * N : (idx_hrr + 1) * N]
    htt    = test_vars_values[idx_htt * N : (idx_htt + 1) * N]
    hpp    = test_vars_values[idx_hpp * N : (idx_hpp + 1) * N]
    dhrrdx     = get_dfdx(hrr, oneoverdx)
    dhttdx     = get_dfdx(htt, oneoverdx)
    dhppdx     = get_dfdx(hpp, oneoverdx)
    
    # assign lambdar values
    for ix in range(num_ghosts, N-num_ghosts) :

        # position on the grid
        r_here = r[ix]
        
        # Assign BSSN vars to local tensors
        h = np.zeros_like(rank_2_spatial_tensor)
        h[i_r][i_r] = hrr[ix]
        h[i_t][i_t] = htt[ix]
        h[i_p][i_p] = hpp[ix]
        
        dhdr = np.zeros_like(rank_2_spatial_tensor)
        dhdr[i_r][i_r] = dhrrdx[ix]
        dhdr[i_t][i_t] = dhttdx[ix]
        dhdr[i_p][i_p] = dhppdx[ix]
        
        # (unscaled) \bar\gamma_ij and \bar\gamma^ij
        bar_gamma_LL = get_metric(r_here, h)
        bar_gamma_UU = get_inverse_metric(r_here, h)
        
        # The connections Delta^i, Delta^i_jk and Delta_ijk
        Delta_U, Delta_ULL, Delta_LLL  = get_connection(r_here, bar_gamma_UU, bar_gamma_LL, h, dhdr)
        test_vars_values[ix + idx_lambdar * N]   = Delta_U[i_r]

    # Fill boundary cells for lambdar
    boundary_cells = np.array([(idx_lambdar + 1)*N-3, (idx_lambdar + 1)*N-2, (idx_lambdar + 1)*N-1])
    for count, ix in enumerate(boundary_cells) :
        offset = -1 - count
        test_vars_values[ix]    = test_vars_values[ix + offset]
        
    boundary_cells = np.array([(idx_lambdar)*N, (idx_lambdar)*N+1, (idx_lambdar)*N+2])
    for count, ix in enumerate(boundary_cells) :
        offset = 5 - 2*count
        test_vars_values[ix] = test_vars_values[ix + offset] * parity[idx_lambdar]
        
    return r, test_vars_values

# This routine gives us the Schwazschild metric in ingoing eddington finkelstien coords (t = t_schwarz - (r-r*))
def get_test_vars_values_bh(R, N_r) :

    # Set up grid values and params
    dx = R/N_r
    N = N_r + num_ghosts * 2 
    r = np.linspace(-(num_ghosts-0.5)*dx, R+(num_ghosts-0.5)*dx, N)
    oneoverdx  = 1.0 / dx
    oneoverdxsquared = oneoverdx * oneoverdx
    GM = 1.0
    
    test_vars_values = np.zeros(NUM_VARS * N)
    
    # Use the Kerr Schild solution with a=0 which has non trivial Kij
    for ix in range(num_ghosts, N-num_ghosts) :

        # position on the grid
        r_i = r[ix]

        # scalar field values
        test_vars_values[ix + idx_u * N] = 0.0
        test_vars_values[ix + idx_v * N] = 0.0
 
        # non zero metric variables (note h_rr etc are rescaled difference from flat space so zero
        # and conformal factor is zero for flat space)
        # note that we choose that the determinant \bar{gamma} = \hat{gamma} initially
        H = 2.0 * GM / r_i
        dHdr = - 2.0 * GM / r_i / r_i
        grr = 1.0 + H
        gtt_over_r2 = 1.0
        # The following is required for spherical symmetry so should not be changed
        gpp_over_r2sintheta = gtt_over_r2
        phys_gamma_over_r4sin2theta = grr * gtt_over_r2 * gpp_over_r2sintheta
        phi_here = 1.0/12.0 * np.log(phys_gamma_over_r4sin2theta)
        test_vars_values[ix + idx_phi * N]   = phi_here
        em4phi = np.exp(-4.0*phi_here)
        test_vars_values[ix + idx_hrr * N]   = em4phi * grr - 1.0
        test_vars_values[ix + idx_htt * N]   = em4phi * gtt_over_r2 - 1.0
        test_vars_values[ix + idx_hpp * N]   = em4phi * gpp_over_r2sintheta - 1.0
        
        # set non zero lapse, shift
        lapse = 1.0/np.sqrt(1.0 + H)
        test_vars_values[ix + idx_lapse * N]   = lapse
        test_vars_values[ix + idx_shiftr * N]   = H / grr
        
        # These are \Gamma^i_jk
        chris_rrr = 0.5 * dHdr / grr
        chris_rtt = - r_i / grr
        chris_rpp = chris_rtt * sin2theta
        
        #K_ij = (D_i shift_j + D_j shift_i) / lapse / 2 (since dgammadt = 0)        
        Krr = (dHdr - chris_rrr * H) / lapse
        Ktt_over_r2 = - chris_rtt * H / lapse / r_i / r_i # 2.0 * lapse / r_i / r_i
        Kpp_over_r2sintheta = - chris_rpp * H / lapse / r_i / r_i / sin2theta
        K = Krr / grr + Ktt_over_r2 / gtt_over_r2  + Kpp_over_r2sintheta / gpp_over_r2sintheta
        test_vars_values[ix + idx_arr * N]   = em4phi * (Krr - 1.0/3.0 * grr * K)
        test_vars_values[ix + idx_att * N]   = em4phi * (Ktt_over_r2 - 1.0/3.0 * gtt_over_r2 * K)
        test_vars_values[ix + idx_app * N]   = em4phi * (Kpp_over_r2sintheta - 1.0/3.0 * gpp_over_r2sintheta * K)
        test_vars_values[ix + idx_K * N]     = K
        
    # overwrite outer boundaries with extrapolation (zeroth order)
    for ivar in range(0, NUM_VARS) :
        boundary_cells = np.array([(ivar + 1)*N-3, (ivar + 1)*N-2, (ivar + 1)*N-1])
        for count, ix in enumerate(boundary_cells) :
            offset = -1 - count
            test_vars_values[ix]    = test_vars_values[ix + offset]

    # overwrite inner cells using parity under r -> - r
    for ivar in range(0, NUM_VARS) :
        boundary_cells = np.array([(ivar)*N, (ivar)*N+1, (ivar)*N+2])
        var_parity = parity[ivar]
        for count, ix in enumerate(boundary_cells) :
            offset = 5 - 2*count
            test_vars_values[ix] = test_vars_values[ix + offset] * var_parity           

    # needed for lambdar
    hrr    = test_vars_values[idx_hrr * N : (idx_hrr + 1) * N]
    htt    = test_vars_values[idx_htt * N : (idx_htt + 1) * N]
    hpp    = test_vars_values[idx_hpp * N : (idx_hpp + 1) * N]
    dhrrdx     = get_dfdx(hrr, oneoverdx)
    dhttdx     = get_dfdx(htt, oneoverdx)
    dhppdx     = get_dfdx(hpp, oneoverdx)
    
    # assign lambdar values
    for ix in range(num_ghosts, N-num_ghosts) :

        # position on the grid
        r_here = r[ix]
        
        # Assign BSSN vars to local tensors
        h = np.zeros_like(rank_2_spatial_tensor)
        h[i_r][i_r] = hrr[ix]
        h[i_t][i_t] = htt[ix]
        h[i_p][i_p] = hpp[ix]
        
        dhdr = np.zeros_like(rank_2_spatial_tensor)
        dhdr[i_r][i_r] = dhrrdx[ix]
        dhdr[i_t][i_t] = dhttdx[ix]
        dhdr[i_p][i_p] = dhppdx[ix]
        
        # (unscaled) \bar\gamma_ij and \bar\gamma^ij
        bar_gamma_LL = get_metric(r_here, h)
        bar_gamma_UU = get_inverse_metric(r_here, h)
        
        # The connections Delta^i, Delta^i_jk and Delta_ijk
        Delta_U, Delta_ULL, Delta_LLL  = get_connection(r_here, bar_gamma_UU, bar_gamma_LL, h, dhdr)
        test_vars_values[ix + idx_lambdar * N]   = Delta_U[i_r]

    # Fill boundary cells for lambdar
    boundary_cells = np.array([(idx_lambdar + 1)*N-3, (idx_lambdar + 1)*N-2, (idx_lambdar + 1)*N-1])
    for count, ix in enumerate(boundary_cells) :
        offset = -1 - count
        test_vars_values[ix]    = test_vars_values[ix + offset]
        
    boundary_cells = np.array([(idx_lambdar)*N, (idx_lambdar)*N+1, (idx_lambdar)*N+2])
    for count, ix in enumerate(boundary_cells) :
        offset = 5 - 2*count
        test_vars_values[ix] = test_vars_values[ix + offset] * parity[idx_lambdar]
        
    return r, test_vars_values
