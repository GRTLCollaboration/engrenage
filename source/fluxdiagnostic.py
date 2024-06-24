#fluxdiagnostics.py

# python modules
import numpy as np
import time

# homemade code
from source.uservariables import *
from source.Grid import *
from source.tensoralgebra import *
from source.mymatter import *

# The diagnostic function returns the flux as a function of r over the grid
# it takes in the solution of the evolution, which is the state vector at every
# time step, and returns the spatial profile F(r) at each time step
def get_Flux_diagnostic(solutions_over_time, t, my_grid) :

    start = time.time()
    
    # For readability
    r = my_grid.r_vector
    N = my_grid.num_points_r
    
    flux = []
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
        
        # Matter sources
        matter_Si  = get_Si(u, dudx, v)

        # End of: Calculate some useful quantities, now start diagnostic
        #################################################################

        # First raise the index on r
        Sr_up =  em4phi * bar_gamma_UU[i_r][:] * matter_Si[i_r]
        sqrt_det_gamma = np.exp(6.0*phi) * r * r
        
        # Get the flux F = 4\pi sqrt(det_gamma) S^r
        flux_i = Sr_up * 4.0 * np.pi * sqrt_det_gamma
        
        # Add the flux value to the output
        flux.append(flux_i)
        
    # end of iteration over time  
    #########################################################################
    
    end = time.time()
    #print("time at t= ", t_i, " is, ", end-start)
    
    return flux
