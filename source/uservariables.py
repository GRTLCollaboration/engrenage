#uservariables.py

# This file provides the list of (rescaled) variables to be evolved and
# assigns each one an index and its parity
# For description of the data structure see https://github.com/GRChombo/engrenage/wiki/Useful-code-background

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

variable_names = ["u", "v", "phi", "hrr", "htt", "hpp", 
                  "K", "arr", "att", "app", 
                  "lambdar", "shiftr", "br", "lapse"]

# parity under r -> -r
parity = [1, 1,           # u, v
          1, 1, 1, 1,     # phi, h
          1, 1, 1, 1,     # K, a
          -1, -1, -1, 1]  # lambda^r, shift^r, b^r, lapse

# scaling at larger r as power of r, i.e. var ~ r^asymptotic_power
asymptotic_power =   [0., 0.,                # u, v
                      -1., -1., -1., -1.,    # phi, h
                      -2., -2., -2., -2.,    # K, a
                      -2., -1., -1., 0.]     # lambda^r, shift^r, b^r, lapse