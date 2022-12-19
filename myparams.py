# myparams.py

# specify the params that are fixed throughout the evolution

import numpy as np

# Input parameters for grid and evolution here
N_r = 50 # num points on physical grid
R = 50.0 # Maximum outer radius

# coefficients for bssn and gauge evolution
eta = 2.0 # 1+log slicing damping coefficient of order 1/M_adm of spacetime
sigma = 1.0 # kreiss-oliger damping coefficient
eight_pi_G = 8.0 * np.pi * 1.0 # Newtons constant, we take G=c=1
scalar_mu = 1.0 # this is an inverse length scale related to the scalar compton wavelength

# These values are hardcoded or calculated from the inputs above
# so should not be changed
dx = R/N_r
num_ghosts = 3
N = N_r + num_ghosts * 2 
r = np.linspace(-(num_ghosts-0.5)*dx, R+(num_ghosts-0.5)*dx, N)
oneoverdx  = 1.0 / dx
oneoverdxsquared = oneoverdx * oneoverdx