"""
This module defines all the bssn variables in the state vector
It provides the list of (rescaled) state variables to be evolved and assigns each one an index and its parity.
For description of the data structure see https://github.com/GRChombo/engrenage/wiki/Useful-code-background.
"""

import numpy as np

__all__ = [
    "idx_phi", "idx_hrr", "idx_htt", "idx_hpp",
    "idx_K", "idx_arr", "idx_att", "idx_app",
    "idx_lambdar", "idx_shiftr", "idx_br", "idx_lapse",
    "BSSN_PARITY", "BSSN_ASYMP_POWER", "BSSN_ASYMP_OFFSET", "NUM_BSSN_VARS",
    "BSSN_VARIABLE_NAMES",
]

idx_phi     = 0    # conformal factor of metric, \gamma_ij = e^{4 \phi} \bar \gamma_ij
idx_hrr     = 1    # rescaled \epsilon_rr -> h_rr - deviation of rr component of the metric from flat
idx_htt     = 2    # rescaled \epsilon_tt -> h_tt - deviation of tt component of the metric from flat
idx_hpp     = 3    # rescaled \epsilon_pp -> h_pp - deviation of pp component of the metric from flat
idx_K       = 4    # mean curvature K
idx_arr     = 5    # rescaled \tilde A_rr -> a_rr - (roughly) time derivative of hrr
idx_att     = 6    # rescaled \tilde A_tt -> a_tt - (roughly) time derivative of htt
idx_app     = 7    # rescaled \tilde A_pp -> a_pp - (roughly) time derivative of hpp
idx_lambdar = 8   # rescaled \bar\Lambda -> lambda^r 
idx_shiftr  = 9   # rescaled \beta^r -> radial shift - gauge variable for relabelling spatial points
idx_br      = 10   # rescaled B^r -> b^r - time derivative of shift
idx_lapse   = 11   # lapse - gauge variable for time slicing

NUM_BSSN_VARS = idx_lapse + 1

BSSN_VARIABLE_NAMES = ["phi", "hrr", "htt", "hpp",
                  "K", "arr", "att", "app", 
                  "lambdar", "shiftr", "br", "lapse"]

# parity under r -> -r
BSSN_PARITY = np.array(
       [1, 1, 1, 1,             # phi, h
        1, 1, 1, 1,             # K, a
        -1, -1, -1, 1]          # lambda^r, shift^r, b^r, lapse
)

# scaling at larger r as power of r, i.e. var ~ r^asymptotic_power + asymptotic_offset
BSSN_ASYMP_POWER = np.array(
       [-1., -1., -1., -1.,     # phi, h
        -1., -2., -2., -2.,     # K, a
        -2., -1., -1., 0.]      # lambda^r, shift^r, b^r, lapse
)

BSSN_ASYMP_OFFSET = np.array(
         [0, 0, 0, 0,            # phi, h
         0, 0, 0, 0,            # K, a
         0, 0, 0, 1]            # lambda^r, shift^r, b^r, lapse
)

