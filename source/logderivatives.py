#logderivatives.py

import numpy as np

# function that returns the derivatives of a field f in 1D (radial), assuming that the grid is
# structured with the first point at dr/2 and subsequent points at dr/2 + c*dr, dr/2 + c*c*dr etc
# note that for consistency the inner ghost cells fall at -dr/2, -dr/2 - dr/c, -dr/2 - dr/c/c
# For description of the grid setup see https://github.com/KAClough/BabyGRChombo/wiki/Useful-code-background

# Assumes num_ghosts ghost cells at either end of the vector of values
from source.uservariables import *

#Logarithmic factor c and related factors
c = 1.075
c2 = c*c
c3 = c2 * c
c4 = c2 * c2
c7 = c3 * c4
c8 = c4 * c4
oneplusc = 1.0 + c
oneplusc2 = 1.0 + c2
onepluscplusc2 = 1.0 + c + c*c

# Centered first derivative (fourth order)
Ap2 = - 1.0 / ( c2 * oneplusc * oneplusc2 * onepluscplusc2 );
Ap1 = oneplusc / (c2 * onepluscplusc2 );
A0 = 2.0 * (c - 1.0) / c;
Am1 = - c4 * Ap1;
Am2 = - c8 * Ap2;

# Centered second derivative (third order)
Bp2 = 2.0 * (1.0 - 2.0*c2 ) / ( c3 * oneplusc * oneplusc * oneplusc2 * onepluscplusc2 );
Bp1 = 2.0 * (2.0*c - 1.0) * oneplusc / ( c3 * onepluscplusc2 );
B0  = 2.0 * (1.0 - c - 5.0*c2 - c3 + c4) / ( c2 * oneplusc * oneplusc ); 
Bm1 = 2.0 * (2.0 - c) * c * oneplusc / onepluscplusc2;
Bm2 = 2.0 * c7 * (c2 - 2.0) / ( c2 * oneplusc * oneplusc * oneplusc2 * onepluscplusc2 );

# downwind (right) first derivative (third order)
Cp2 = - 1.0 / ( c2 * oneplusc * onepluscplusc2 );
Cp1 = 1.0 / c2;
C0 = ( c2 - 2.0 ) / (c * oneplusc);
Cm1 = - c2 / onepluscplusc2;

# upwind (left) first derivative (third order)
Dp1 = 1.0 / ( c * onepluscplusc2 );
D0 = ( 2.0 * c2 - 1.0) / ( c * oneplusc );
Dm1 = - c;
Dm2 = c4 / ( oneplusc * onepluscplusc2 );

# second derivative
# Note that convolution inverts the stencils so it is the opposite of the order above
d2dx2_stencil = np.array([Bp2, Bp1, B0, Bm1, Bm2])

def get_logd2fdx2(f, oneoverdrsquared) :
    # Convolve with the stencil; mode='same' will give result of size of f
    f_xx = np.convolve(f, d2dx2_stencil, mode='same')
    # Clear out the ghost zones
    f_xx[0:num_ghosts] = 0.
    f_xx[-num_ghosts:] = 0.

    return oneoverdrsquared * f_xx

# first derivative
# Note that convolution inverts the stencils so it is the opposite of the order above
ddx_stencil = np.array([Ap2, Ap1, A0, Am1, Am2])

def get_logdfdx(f, oneoverdr) :
    # Convolve with the stencil; mode='same' will give result of size of f
    f_x = np.convolve(f, ddx_stencil, mode='same')
    # Clear out the ghost zones
    f_x[0:num_ghosts] = 0.
    f_x[-num_ghosts:] = 0.
        
    return oneoverdr * f_x

# advective derivatives
# Note that convolution inverts the stencils so these are the opposite of the order above
ddx_stencil_left  = np.array([ 0,  Dp1, D0, Dm1, Dm2])
ddx_stencil_right = np.array([ Cp2,  Cp1, C0, Cm1, 0.])

def get_logdfdx_advec_L(f, oneoverdr) :
    f_xL = np.convolve(f, ddx_stencil_left, mode='same')       
        
    # Clear out the ghost zones
    f_xL[0:num_ghosts] = 0.
    f_xL[-num_ghosts:] = 0.
        
    return oneoverdr * f_xL

def get_logdfdx_advec_R(f, oneoverdr) :
    f_xR = np.convolve(f, ddx_stencil_right, mode='same')        
        
    # Clear out the ghost zones
    f_xR[0:num_ghosts] = 0.
    f_xR[-num_ghosts:] = 0.
        
    return oneoverdr * f_xR

# 2N = 4 Kreiss Oliger dissipation
# Copied from Thomas's code, need to check derivation, why no oneoverdr?
# diss_stencil = np.array([1.0, +1.0, 6.0, -4.0, 1.0])
diss_stencil = np.array([+1., -6., +15., -20., +15., -6., +1.]) / 64.0

def get_logdissipation(f, oneoverdr, sigma) :
    diss_x = np.convolve(f, diss_stencil, mode='same')  
        
    # Clear out the ghost zones and zero near outer boundary
    diss_x[0:num_ghosts] = 0.
    diss_x[-(num_ghosts+3):] = 0.
        
    return sigma * diss_x * oneoverdr

