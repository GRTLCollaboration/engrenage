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

# parity under r -> -r
parity = [1, 1,           # u, v
          1, 1, 1, 1,     # phi, h
          1, 1, 1, 1,     # K, a
          -1, -1, -1, 1]  # lambda^r, shift^r, b^r, lapse

# scaling at larger r as power of r, i.e. var ~ r^asymptotic_power
asymptotic_power =   [0., 0.,                # u, v
                      -1., -1., -1., -1.,    # phi, h
                      0., 0., 0., 0., #-2., -2., -2., -2.,    # K, a
                      0., 0., 0., 0.] #-2., -1., -1., 0.]     # lambda^r, shift^r, b^r, lapse

# hard code number of ghosts to 3 here
num_ghosts = 3

def unpack_vars_vector(vars_vec, N_r) :

    domain_length = N_r + 2 * num_ghosts
    
    # Scalar field vars
    u    = vars_vec[idx_u * domain_length : (idx_u + 1) * domain_length]
    v    = vars_vec[idx_v * domain_length : (idx_v + 1) * domain_length]
    
    # Conformal factor and rescaled perturbation to spatial metric
    phi    = vars_vec[idx_phi * domain_length : (idx_phi + 1) * domain_length]
    hrr    = vars_vec[idx_hrr * domain_length : (idx_hrr + 1) * domain_length]
    htt    = vars_vec[idx_htt * domain_length : (idx_htt + 1) * domain_length]
    hpp    = vars_vec[idx_hpp * domain_length : (idx_hpp + 1) * domain_length]

    # Mean curvature and rescaled perturbation to traceless A_ij
    K      = vars_vec[idx_K   * domain_length : (idx_K   + 1) * domain_length]
    arr    = vars_vec[idx_arr * domain_length : (idx_arr + 1) * domain_length]
    att    = vars_vec[idx_att * domain_length : (idx_att + 1) * domain_length]
    app    = vars_vec[idx_app * domain_length : (idx_app + 1) * domain_length]

    # Gamma^x, shift and lapse
    lambdar    = vars_vec[idx_lambdar * domain_length : (idx_lambdar + 1) * domain_length]
    shiftr     = vars_vec[idx_shiftr  * domain_length : (idx_shiftr  + 1) * domain_length]
    br         = vars_vec[idx_br      * domain_length : (idx_br      + 1) * domain_length]
    lapse      = vars_vec[idx_lapse   * domain_length : (idx_lapse   + 1) * domain_length]    
    
    return u, v , phi, hrr, htt, hpp, K, arr, att, app, lambdar, shiftr, br, lapse
