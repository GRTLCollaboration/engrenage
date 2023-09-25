#gridfunctions.py

import numpy as np
from source.uservariables import *

# For description of the grid setup see https://github.com/GRChombo/engrenage/wiki/Useful-code-background

# hard code number of ghosts to 3 here
num_ghosts = 3

# fix the value of c for logarithmic grid and precompute related functions of it
c = 1.075
c2 = c*c
c3 = c2 * c
c4 = c2 * c2
c7 = c3 * c4
c8 = c4 * c4
oneplusc = 1.0 + c
oneplusc2 = 1.0 + c2
onepluscplusc2 = 1.0 + c + c*c

# Finite difference coefficients

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

# Set up the grid
def setup_grid(R, N_r, r_is_logarithmic) :
    
    # For a linear grid
    dx = R / N_r
    N = N_r + num_ghosts * 2 
    r = np.linspace(-(num_ghosts-0.5)*dx, R+(num_ghosts-0.5)*dx, N)
    logarithmic_dr = np.ones_like(r)
    
    if (r_is_logarithmic) :
        # overwrite grid values for logarithmic grid
        # We want the domain outer R to be at R, so... 
        dx = R * (c-1) / (c ** N_r - 1)
        logarithmic_dr[num_ghosts] = dx
        logarithmic_dr[num_ghosts-1] = logarithmic_dr[num_ghosts]/c
        logarithmic_dr[num_ghosts-2] = logarithmic_dr[num_ghosts-1]/c
        logarithmic_dr[num_ghosts-3] = logarithmic_dr[num_ghosts-2]/c        
        r[num_ghosts] = dx / 2.0
        r[num_ghosts - 1] = - dx / 2.0
        r[num_ghosts - 2] = r[num_ghosts - 1] - dx / c
        r[num_ghosts - 3] = r[num_ghosts - 2] - dx / c / c
        for idx in np.arange(num_ghosts, N, 1) :
            logarithmic_dr[idx] = logarithmic_dr[idx-1] * c
            r[idx] = r[idx-1] + logarithmic_dr[idx]
    
    return dx, N, r, logarithmic_dr

# fills the inner boundary ghosts at r=0 end
def fill_inner_boundary(state, dx, N, r_is_logarithmic) :
    
    for ivar in range(0, NUM_VARS) :
        fill_inner_boundary_ivar(state, dx, N, r_is_logarithmic, ivar)
                
def fill_inner_boundary_ivar(state, dx, N, r_is_logarithmic, ivar) :

    var_parity = parity[ivar]
    if (r_is_logarithmic) :
        dist1 = dx / c - c * dx # distance to ghost element -2
        dist2 = dx / c + dx / c / c - c * dx # distance to ghost element -3
        oneoverlogdr_a = 1.0 / (dx * c)
        oneoverlogdr2_a = oneoverlogdr_a * oneoverlogdr_a        
        idx_a = ivar * N + num_ghosts + 1 # Point a is the second valid point in the grid above r=0
        # first impose the symmetry about zero for ghost element -1
        state[idx_a - 2] = state[idx_a - 1] * var_parity
        # calculate gradients at a
        dfdx_a   = (Am2 * state[idx_a-2] + Am1 * state[idx_a-1] + A0 * state[idx_a] 
                  + Ap1 * state[idx_a+1] + Ap2 * state[idx_a+2]) * oneoverlogdr_a
        d2fdx2_a = (Bm2 * state[idx_a-2] + Bm1 * state[idx_a-1] + B0 * state[idx_a] 
                  + Bp1 * state[idx_a+1] + Bp2 * state[idx_a+2]) * oneoverlogdr2_a
        # Use taylor series approximation to fill points
        state[idx_a - 3] = (state[idx_a] + dist1 * dfdx_a
                              + 0.5 * (dist1 * dist1) * d2fdx2_a ) * var_parity
        state[idx_a - 4] = (state[idx_a] + dist2 * dfdx_a
                              + 0.5 * (dist2 * dist2) * d2fdx2_a ) * var_parity            
    else : 
        # Apply a simple reflection of the values
        boundary_cells = np.array([(ivar)*N, (ivar)*N+1, (ivar)*N+2])
        for count, ix in enumerate(boundary_cells) :
            offset = 5 - 2*count
            state[ix] = state[ix + offset] * var_parity

# fills the outer boundary ghosts at large r
def fill_outer_boundary(state, dx, N, r_is_logarithmic) :

    for ivar in range(0, NUM_VARS) :
        fill_outer_boundary_ivar(state, dx, N, r_is_logarithmic, ivar)
    
def fill_outer_boundary_ivar(state, dx, N, r_is_logarithmic, ivar) :
    
    R_lin = dx * (N - 2 * num_ghosts)
    r_linear = np.linspace(-(num_ghosts-0.5)*dx, R_lin+(num_ghosts-0.5)*dx, N)    
    boundary_cells = np.array([(ivar + 1)*N-3, (ivar + 1)*N-2, (ivar + 1)*N-1])
    for count, ix in enumerate(boundary_cells) :
        offset = -1 - count
        if(r_is_logarithmic) :
            #zeroth order interpolation for now
            state[ix]    = state[ix + offset]            
        else :
            # use asymptotic powers
            state[ix]    = state[ix + offset] * ((r_linear[N - 3 + count] / r_linear[N - 4]) 
                                                                         ** asymptotic_power[ivar])
