#uservariables.py

# This file provides the list of (rescaled) variables to be evolved and
# assigns each one an index and its parity

idx_u       = 0    # scalar field
idx_v       = 1    # scalar field conjugate momentum
idx_phi     = 2    # conformal factor of metric
idx_hrr     = 3    # rescaled \epsilon_rr -> h_rr
idx_htt     = 4    # rescaled \epsilon_tt -> h_tt
idx_hpp     = 5    # rescaled \epsilon_pp -> h_pp
idx_K       = 6    # mean curvature K
idx_arr     = 7    # rescaled \tilde A_rr -> a_rr
idx_att     = 8    # rescaled \tilde A_tt -> a_tt
idx_app     = 9    # rescaled \tilde A_pp -> a_pp
idx_lambdar = 10   # rescaled \bar\Lambda -> lambda_rr
idx_shiftr  = 11   # rescaled \beta^r -> shift_r
idx_br      = 12   # rescaled B^r -> b_r
idx_lapse   = 13   # lapse

NUM_VARS = idx_lapse + 1

parity = [1, 1,           # u, v
          1, 1, 1, 1,     # phi, h
          1, 1, 1, 1,     # K, a
          -1, -1, -1, 1]  # lambda^r, shift^r, b^r, lapse
          
