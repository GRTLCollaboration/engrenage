#fourthorderderivatives.py

import numpy as np
from myparams import *

# function that returns the derivatives of a field f in 1D (radial)
# Assumes num_ghosts ghost cells at either end of the vector of values of length NN

# second derivative

d2dx2_stencil = oneoverdxsquared * np.array([-1., +16., -30., +16., -1.0]) / 12.0

def get_d2fdx2(f) :
    # Convolve with the stencil; mode='same' will give result of size of f
    f_xx = np.convolve(f, d2dx2_stencil, mode='same')
    # Clear out the ghost zones
    f_xx[0:num_ghosts] = 0.
    f_xx[-num_ghosts:] = 0.

    return f_xx


ddx_stencil = oneoverdx * np.array([-1., +8., 0., -8., +1.]) / 12.0

# first derivative
def get_dfdx(f) :
    # Convolve with the stencil; mode='same' will give result of size of f
    f_x = np.convolve(f, ddx_stencil, mode='same')
    # Clear out the ghost zones
    f_x[0:num_ghosts] = 0.
    f_x[-num_ghosts:] = 0.
        
    return f_x

# 2N = 6 Kreiss Oliger dissipation
def get_dissipation(f) :
    diss_x = np.zeros_like(f)
    NN = np.size(f)
    for i, f_i in enumerate(f) :

        if (i > (num_ghosts-1) and i < NN-num_ghosts) :
        
            # indices of neighbouring points
            i_m3 = i-3
            i_m2 = i-2
            i_m1 = i-1
            i_p1 = i+1
            i_p2 = i+2
            i_p3 = i+3
            
            diss_x[i] = sigma * 1./64.0 * oneoverdx * ( 
                                              + 1.0  * f[i_m3]
                                              - 6.0  * f[i_m2] 
                                              + 15.0 * f[i_m1] 
                                              - 20.0 * f[i]
                                              + 15.0 * f[i_p1]
                                              - 6.0  * f[i_p2]
                                              + 1.0  * f[i_p3]  )
        
    return diss_x

