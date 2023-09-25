#logderivatives.py

import numpy as np
from source.gridfunctions import *

# function that returns the derivatives of a field f in 1D (radial), assuming that the grid is
# structured with the first point at dr/2 and subsequent points at dr/2 + c*dr, dr/2 + c*c*dr etc
# note that for consistency the inner ghost cells fall at -dr/2, -dr/2 - dr/c, -dr/2 - dr/c/c
# For description of the grid setup see https://github.com/GRChombo/engrenage/wiki/Useful-code-background
# Note that these fill the whole r vector of values but the boundary values will be overwritten
# The coefficient values here are specified in gridfunctions

# second derivative
# Note that convolution inverts the stencils so it is the opposite of the order you expect
d2dx2_stencil = np.array([Bp2, Bp1, B0, Bm1, Bm2])

def get_logd2fdx2(f, oneoverdrsquared) :
    
    # Convolve with the stencil; mode='same' will give result of size of f
    f_xx = np.convolve(f, d2dx2_stencil, mode='same')

    return oneoverdrsquared * f_xx

# first derivative
# Note that convolution inverts the stencils so it is the opposite of the order you expect
ddx_stencil = np.array([Ap2, Ap1, A0, Am1, Am2])

def get_logdfdx(f, oneoverdr) :
    
    # Convolve with the stencil; mode='same' will give result of size of f
    f_x = np.convolve(f, ddx_stencil, mode='same')
        
    return oneoverdr * f_x

# advective derivatives
# Note that convolution inverts the stencils so these are the opposite of the order you expect
ddx_stencil_left  = np.array([ 0,  Dp1, D0, Dm1, Dm2])
ddx_stencil_right = np.array([ Cp2,  Cp1, C0, Cm1, 0.])

def get_logdfdx_advec_L(f, oneoverdr) :

    # Convolve with the stencil; mode='same' will give result of size of f
    f_xL = np.convolve(f, ddx_stencil_left, mode='same')       
    
    return oneoverdr * f_xL

def get_logdfdx_advec_R(f, oneoverdr) :

    # Convolve with the stencil; mode='same' will give result of size of f
    f_xR = np.convolve(f, ddx_stencil_right, mode='same')        
        
    return oneoverdr * f_xR

# 2N = 4 Kreiss Oliger dissipation
# For now just use the regular stencil for linear dissipation as this
# seems to work and should be ok if c is roughly 1 (we multiply by the dr too).
diss_stencil = np.array([+1., -6., +15., -20., +15., -6., +1.]) / 64.0

def get_logdissipation(f, oneoverdr, sigma) :
    
    # Convolve with the stencil; mode='same' will give result of size of f    
    diss_x = np.convolve(f, diss_stencil, mode='same')  
        
    return sigma * diss_x * oneoverdr

