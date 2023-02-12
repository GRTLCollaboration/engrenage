#fourthorderderivatives.py

import numpy as np

# function that returns the derivatives of a field f in 1D (radial)
# For description of the grid setup see https://github.com/KAClough/BabyGRChombo/wiki/Useful-code-background

# Assumes num_ghosts ghost cells at either end of the vector of values
from source.gridfunctions import *

# second derivative
d2dx2_stencil = np.array([-1., +16., -30., +16., -1.0]) / 12.0

def get_d2fdx2(f, oneoverdxsquared) :
    # Convolve with the stencil; mode='same' will give result of size of f
    f_xx = np.convolve(f, d2dx2_stencil, mode='same')
    # Clear out the ghost zones
    f_xx[0:num_ghosts] = 0.
    f_xx[-num_ghosts:] = 0.

    return oneoverdxsquared * f_xx

# first derivative
# Note that convolution inverts the stencils so it is the opposite of the order here:
# https://web.media.mit.edu/~crtaylor/calculator.html
ddx_stencil = np.array([-1., +8., 0., -8., +1.]) / 12.0

def get_dfdx(f, oneoverdx) :
    # Convolve with the stencil; mode='same' will give result of size of f
    f_x = np.convolve(f, ddx_stencil, mode='same')
    # Clear out the ghost zones
    f_x[0:num_ghosts] = 0.
    f_x[-num_ghosts:] = 0.
        
    return oneoverdx * f_x

# advective derivatives
# Note that convolution inverts the stencils so these are the opposite of the order here:
# https://web.media.mit.edu/~crtaylor/calculator.html
ddx_stencil_left  = np.array([ 0., 0., +3., +10.,  -18.,  +6.,  -1.]) / 12.0
ddx_stencil_right = np.array([ +1.,  -6.,  +18., -10., -3., 0., 0.]) / 12.0

def get_dfdx_advec_L(f, oneoverdx) :
    f_xL = np.convolve(f, ddx_stencil_left, mode='same')       
        
    # Clear out the ghost zones
    f_xL[0:num_ghosts] = 0.
    f_xL[-num_ghosts:] = 0.
        
    return oneoverdx * f_xL

def get_dfdx_advec_R(f, oneoverdx) :
    f_xR = np.convolve(f, ddx_stencil_right, mode='same')        
        
    # Clear out the ghost zones
    f_xR[0:num_ghosts] = 0.
    f_xR[-num_ghosts:] = 0.
        
    return oneoverdx * f_xR

# 2N = 6 Kreiss Oliger dissipation
diss_stencil = np.array([+1., -6., +15., -20., +15., -6., +1.]) / 64.0

def get_dissipation(f, oneoverdx, sigma) :
    diss_x = np.convolve(f, diss_stencil, mode='same')  
        
    # Clear out the ghost zones and zero near outer boundary
    diss_x[0:num_ghosts] = 0.
    diss_x[-(num_ghosts+3):] = 0.
        
    return oneoverdx * sigma * diss_x

