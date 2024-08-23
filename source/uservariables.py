"""
This module defines all the constants used in the code.

It provides the list of (rescaled) variables to be evolved and assigns each one an index and its parity.
For description of the data structure see https://github.com/GRChombo/engrenage/wiki/Useful-code-background.
"""

from enum import Enum

import numpy as np


__all__ = [
    "NUM_GHOSTS", "SPACEDIM",
    # Spatial indices
    "i_r", "i_t", "i_p",
    # Indices of state variables.
    "idx_u", "idx_v",
    "idx_phi", "idx_hrr", "idx_htt", "idx_hpp",
    "idx_K", "idx_arr", "idx_att", "idx_app",
    "idx_lambdar", "idx_shiftr", "idx_br", "idx_lapse",
    "PARITY", "ASYMP_POWER", "ASYMP_OFFSET", "NUM_VARS",
    "SpacingExtent", "VARIABLE_NAMES",
]


NUM_GHOSTS: int = 3

# Spatial indices
SPACEDIM: int = 3
i_r, i_t, i_p = 0, 1, 2


idx_u       = 0    # scalar field
idx_v       = 1    # scalar field conjugate momentum (roughly the time derivative of u)
idx_phi     = 2    # conformal factor of metric, \gamma_ij = e^{4 \phi} \bar \gamma_ij
idx_hrr     = 3    # rescaled \epsilon_rr -> h_rr - deviation of rr component of the metric from flat
idx_htt     = 4    # rescaled \epsilon_tt -> h_tt - deviation of tt component of the metric from flat
idx_hpp     = 5    # rescaled \epsilon_pp -> h_pp - deviation of pp component of the metric from flat
idx_K       = 6    # mean curvature K
idx_arr     = 7    # rescaled \tilde A_rr -> a_rr - (roughly) time derivative of hrr
idx_att     = 8    # rescaled \tilde A_tt -> a_tt - (roughly) time derivative of htt
idx_app     = 9    # rescaled \tilde A_pp -> a_pp - (roughly) time derivative of hpp
idx_lambdar = 10   # rescaled \bar\Lambda -> lambda^r 
idx_shiftr  = 11   # rescaled \beta^r -> radial shift - gauge variable for relabelling spatial points
idx_br      = 12   # rescaled B^r -> b^r - time derivative of shift
idx_lapse   = 13   # lapse - gauge variable for time slicing

NUM_VARS = idx_lapse + 1

VARIABLE_NAMES = ["u", "v", "phi", "hrr", "htt", "hpp",
                  "K", "arr", "att", "app", 
                  "lambdar", "shiftr", "br", "lapse"]


# parity under r -> -r
PARITY = np.array(
        [1, 1,                  # u, v
        1, 1, 1, 1,             # phi, h
        1, 1, 1, 1,             # K, a
        -1, -1, -1, 1]          # lambda^r, shift^r, b^r, lapse
)


# scaling at larger r as power of r, i.e. var ~ r^asymptotic_power + asymptotic_offset
ASYMP_POWER = np.array(
        [0., 0.,                # u, v
        -1., -1., -1., -1.,     # phi, h
        -1., -2., -2., -2.,     # K, a
        -2., -1., -1., 0.]      # lambda^r, shift^r, b^r, lapse
)


ASYMP_OFFSET = np.array(
        [0., 0.,                # u, v
         0, 0, 0, 0,            # phi, h
         0, 0, 0, 0,            # K, a
         0, 0, 0, 1]            # lambda^r, shift^r, b^r, lapse
)


class SpacingExtent(Enum):
    HALF = 0
    FULL = 1
