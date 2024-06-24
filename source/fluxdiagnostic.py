#hamdiagnostics.py

# python modules
import numpy as np
import time

# homemade code
from source.uservariables import *
from source.Grid import *
from source.tensoralgebra import *
from source.mymatter import *

# The diagnostic function returns the Hamiltonian constraint over the grid
# it takes in the solution of the evolution, which is the state vector at every
# time step, and returns the spatial profile Ham(r) at each time step
def get_Ham_diagnostic(solutions_over_time, t, my_grid) :

    start = time.time()
    
    # For readability
    r = my_grid.r_vector
    N = my_grid.num_points_r
    
    Ham = []
    num_times = int(np.size(solutions_over_time) / (NUM_VARS * N))
    
    # unpack the vectors at each time
    for i in range(num_times) :
        
        t_i = t[i]
        
        if(num_times == 1):
            solution = solutions_over_time
        else :
            solution = solutions_over_time[i]

        # Assign the variables to parts of the solution
        (u, v , phi, hrr, htt, hpp, K, 
         arr, att, app, lambdar, shiftr, br, lapse) = np.array_split(solution, NUM_VARS)
        
        ################################################################################################
        
        # first derivatives
        dudx       = np.dot(my_grid.derivatives.d1_matrix, u      )
        dvdx       = np.dot(my_grid.derivatives.d1_matrix, v      )
        dphidx     = np.dot(my_grid.derivatives.d1_matrix, phi    )
        dhrrdx     = np.dot(my_grid.derivatives.d1_matrix, hrr    )
        dhttdx     = np.dot(my_grid.derivatives.d1_matrix, htt    )
        dhppdx     = np.dot(my_grid.derivatives.d1_matrix, hpp    )
        darrdx     = np.dot(my_grid.derivatives.d1_matrix, arr    )
        dattdx     = np.dot(my_grid.derivatives.d1_matrix, att    )
        dappdx     = np.dot(my_grid.derivatives.d1_matrix, app    )
        dKdx       = np.dot(my_grid.derivatives.d1_matrix, K      )
        dlambdardx = np.dot(my_grid.derivatives.d1_matrix, lambdar)            

        # second derivatives
        d2udx2      = np.dot(my_grid.derivatives.d2_matrix, u     )
        d2phidx2    = np.dot(my_grid.derivatives.d2_matrix, phi   )
        d2hrrdx2    = np.dot(my_grid.derivatives.d2_matrix, hrr   )
        d2httdx2    = np.dot(my_grid.derivatives.d2_matrix, htt   )
        d2hppdx2    = np.dot(my_grid.derivatives.d2_matrix, hpp   )      
            
        ##############################################################################################

        h = np.array([hrr, htt, hpp])
        a = np.array([arr, att, app])
        em4phi = np.exp(-4.0*phi)
        dhdr   = np.array([dhrrdx, dhttdx, dhppdx])
        d2hdr2 = np.array([d2hrrdx2, d2httdx2, d2hppdx2])
              
        # Calculate some useful quantities
        ########################################################
        
        # \hat \Gamma^i_jk
        flat_chris = get_flat_spherical_chris(r)
        
        # rescaled \bar\gamma_ij
        r_gamma_LL = get_rescaled_metric(h)
        r_gamma_UU = get_rescaled_inverse_metric(h)
        
        # (unscaled) \bar\gamma_ij and \bar\gamma^ij
        bar_gamma_LL = get_metric(r, h)
        bar_gamma_UU = get_inverse_metric(r, h)
        
        # \bar A_ij, \bar A^ij and the trace A_i^i, then Asquared = \bar A_ij \bar A^ij
        bar_A_LL = get_A_LL(r, a)
        bar_A_UU = get_A_UU(a, r_gamma_UU, r)
        traceA   = get_trace_A(a, r_gamma_UU)
        Asquared = get_Asquared(a, r_gamma_UU)
        
        # The connections Delta^i, Delta^i_jk and Delta_ijk
        Delta_U, Delta_ULL, Delta_LLL  = get_connection(r, bar_gamma_UU, bar_gamma_LL, h, dhdr)
        bar_Rij = get_ricci_tensor(r, h, dhdr, d2hdr2, lambdar, dlambdardx, 
                                       Delta_U, Delta_ULL, Delta_LLL, bar_gamma_UU, bar_gamma_LL)
        bar_Rij_diag = np.array([bar_Rij[i_r][i_r],bar_Rij[i_t][i_t],bar_Rij[i_p][i_p]])
        bar_R   = get_trace(bar_Rij_diag, bar_gamma_UU)
        
        # Matter sources
        matter_rho            = get_rho(u, dudx, v, bar_gamma_UU, em4phi )

        # End of: Calculate some useful quantities, now start diagnostic
        #################################################################

        # Get the Ham constraint eqn (13) of Baumgarte https://arxiv.org/abs/1211.6632
        Ham_i = (  two_thirds * K * K - Asquared
                         + em4phi * ( bar_R
                                      - 8.0 * bar_gamma_UU[i_r][:] * (dphidx * dphidx 
                                                                        + d2phidx2)
               # These terms come from \bar\Gamma^r d_r \phi from the \bar D^2 \phi term
                                      + 8.0 * bar_gamma_UU[i_t][:] 
                                            * flat_chris[i_r][i_t][i_t] * dphidx
                                      + 8.0 * bar_gamma_UU[i_p][:] 
                                            * flat_chris[i_r][i_p][i_p] * dphidx
                                      + 8.0 * Delta_U[i_r] * dphidx)
                         - 2.0 * eight_pi_G * matter_rho ) 
        
        # Add the Ham value to the output
        Ham.append(Ham_i)
        
    # end of iteration over time  
    #########################################################################
    
    end = time.time()
    #print("time at t= ", t_i, " is, ", end-start)
    
    return Ham
