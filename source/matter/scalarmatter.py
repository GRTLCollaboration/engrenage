# import numpy as np
# def safe_exp(x, limit=100.0):
#     x_clipped = np.clip(x, -limit, limit)
#     return np.exp(x_clipped)

# from core.grid import SPACEDIM, i_x1
# from bssn.bssnstatevariables import NUM_BSSN_VARS
# from bssn.tensoralgebra import EMTensor, get_bar_gamma_UU, get_bar_gamma_LL
# from bssn.bssnstatevariables import idx_lapse


# class ScalarMatter:
#     """
#     Massless scalar field for Choptuik critical collapse (spherical symmetry).
#     Implements 3+1 Einstein–scalar without electromagnetic field.
#     """

#     def __init__(self, a_scalar_mu=0.0):
        
#         # Pure collapse: massless scalar (μ = 0)
#         self.scalar_mu       = 0.0
#         self.NUM_MATTER_VARS = 2
#         self.VARIABLE_NAMES  = ["u", "v"]
#         self.PARITY          = np.array([1, 1])
#         self.ASYMP_POWER     = np.array([0, 0])
#         self.ASYMP_OFFSET    = np.array([0, 0])
#         # Indici nel vettore di stato (dopo le variabili BSSN)
#         self.idx_u    = NUM_BSSN_VARS
#         self.idx_v    = NUM_BSSN_VARS + 1
#         self.indices  = np.array([self.idx_u, self.idx_v])
#         self.matter_vars_set = False
        

#     def V_of_u(self, u):
#         # Potenziale nullo
#         return np.zeros_like(u)

#     def dVdu(self, u):
#         return np.zeros_like(u)

#     def set_matter_vars(self, state_vector, bssn_vars, grid):
#         # Estrae θ e Π dal vettore di stato
         
#         self.u = state_vector[self.idx_u].copy()  # θ
#         self.v = state_vector[self.idx_v].copy()  # Π  (se usi v = Π)

#         # Prima derivata radiale ∂_r θ
#         d1 = grid.get_first_derivative(state_vector, [self.idx_u])
#         self.du_dr = np.ravel(d1[self.idx_u])  # shape (N,)
#         self.dv_dr = np.ravel(d1[self.idx_v])  # shape (N,)

#         # Seconda derivata radiale ∂_r² θ
#         d2 = grid.get_second_derivative(state_vector, [self.idx_u])
#         self.d2u_dr2 = np.ravel(d2[self.idx_u])  # shape (N,)
#         #self.d2u_dr2 = np.ravel(d2[self.idx_u])  # shape (N,)
#         # === TEST: confronta con differenze finite manuali ===
#         dr = grid.dr
#         i_test = N//2   # punto circa a metà griglia
        
#         # Derivate numeriche manuali
#         du_fd = (self.u[i_test+1] - self.u[i_test-1]) / (2.0 * dr)
#         d2u_fd = (self.u[i_test+1] - 2*self.u[i_test] + self.u[i_test-1]) / (dr**2)
        
#         print("[TEST] Punto", i_test)
#         print("self.du_dr =", self.du_dr[i_test], " | FD approx =", du_fd)
#         print("self.d2u_dr2 =", self.d2u_dr2[i_test], " | FD approx =", d2u_fd)


#         self.matter_vars_set = True
    

#     def get_emtensor(self, r, bssn_vars, background):
#         assert self.matter_vars_set, "Call set_matter_vars() before get_emtensor()"
#         N = r.size
#         T = EMTensor(N)

#         # Fattori metrici
#         em4phi = safe_exp(-4.0 * bssn_vars.phi)  # shape (N,)
#         bar_uu = get_bar_gamma_UU(r, bssn_vars.h_LL, background)  # (N,3,3)
#         bar_ll = get_bar_gamma_LL(r, bssn_vars.h_LL, background)  # (N,3,3)

#         # Solo componente radiale della metrica invertita
#         g_rr = bar_uu[:, i_x1, i_x1]  # shape (N,)

#         inv16 = 1.0 / (16.0 * np.pi)
#         inv32 = 1.0 / (32.0 * np.pi)

#         # ∂_rθ² term: grad² = em4phi * g^rr * (∂_r θ)²
#         grad2 = em4phi * g_rr * (self.du_dr ** 2)

#         # Componenti del tensore energia‐impulso
#         T.rho = inv16 * (self.v ** 2) + inv32 * grad2 

#         Si = np.zeros((N, SPACEDIM))
#         Si[:, i_x1] = inv16 * self.v * self.du_dr
#         T.Si = Si 

#             # Stress tensor Sij
#         Sij = np.zeros((N, SPACEDIM, SPACEDIM))
    
#         # Termine generale (∂iθ ∂jθ)/(16π)
#         Sij[:, i_x1, i_x1] = inv16 * (self.du_dr ** 2)
    
#         # Termine -γ_ij (∂kθ ∂^kθ)/(32π)
#         for a in range(SPACEDIM):
#             Sij[:, a, a] += -bar_ll[:, a, a] * inv32 * grad2
    
#         T.Sij = Sij

#         # traccia S = γ^{ij} S_ij = em4phi * sum(bar_uu * Sij)
#         T.S = em4phi * np.sum(bar_uu * Sij, axis=(1,2))
        
#         return T
    
    
#     def compute_lap_u(self, r, grid, bssn_vars, background):
#         # Fattori conformi
#         phi    = bssn_vars.phi
#         em6phi = safe_exp(-6.0 * phi)   # e^{-6φ}
#         e2phi  = safe_exp( 2.0 * phi)   # e^{2φ}
    
#         # g^{rr} = e^{-4φ} * bar{γ}^{rr}
#         bar_uu  = get_bar_gamma_UU(r, bssn_vars.h_LL, background)
#         bar_grr = bar_uu[:, i_x1, i_x1]
    
#         du = self.du_dr
    
#         # Quantità dentro la derivata: r^2 e^{2φ} bar{γ}^{rr} ∂_r θ
#         inner = (r**2) * e2phi * bar_grr * du
    
#         # Derivata radiale tramite matrice differenze finite
#         d_inner = grid.derivs.drn_matrix[1] @ inner
    
#         # Laplaciano covariante
#         lap_u = em6phi * (1.0 / (r**2)) * d_inner
    
#         # Regolarizzazione al centro (r=0)
#         lap_u[0] = 3.0 * safe_exp(-4.0 * phi[0]) * bar_grr[0] * self.d2u_dr2[0]
    
#         return lap_u

    

        
#     def get_matter_rhs(self, r, grid, bssn_vars, bssn_d1, background):
#         assert self.matter_vars_set, "Call set_matter_vars() before get_matter_rhs()"

#         # 1) ∂_t θ = L_β θ - α Π
#         dudt = + bssn_vars.shift_U[:, i_x1] * self.du_dr - bssn_vars.lapse * self.v  # shape (N,)

        
#         # 2) Laplaciano sferico
#         lap_u = self.compute_lap_u(r, grid, bssn_vars, background)



#         # 3) ∂^i α ∂_i θ
        
#         alpha_grad_r = bssn_d1[idx_lapse, :]    # ∂_r α
#         grad_alpha   = alpha_grad_r * self.du_dr
#         grad_alpha[0] = 0.0
        
#         em4phi = safe_exp(-4.0 * bssn_vars.phi)
#         bar_uu = get_bar_gamma_UU(r, bssn_vars.h_LL, background)
#         g_rr   = bar_uu[:, i_x1, i_x1]
        
    
#         dvdt = (
#             bssn_vars.shift_U[:, i_x1] * self.dv_dr
#             - bssn_vars.lapse * lap_u
#             - em4phi * g_rr * grad_alpha
#             + bssn_vars.lapse * bssn_vars.K * self.v
#         )
        

#         return dudt, dvdt
# # 2) Laplaciano sferico Δθ = ∂_r²θ + (2/r) ∂_rθ
#         #r_arr = np.ravel(r)  # shape (N,)
#         #lap = np.zeros_like(self.d2u_dr2)
#         #with np.errstate(divide='ignore', invalid='ignore'):
#            # lap = self.d2u_dr2 + (2.0 / r_arr) * self.du_dr
#         # Correggi il centro
#         #lap[0] = self.d2u_dr2[0]

#          # L_β Π = + bssn_vars.shift_U[:, i_x1] * self.dPi_dr
#  # ==============================
#     # DEBUG: stampo solo al primo step
#     # ==============================
#       #  if np.isnan(lap_u).any() or np.isnan(dvdt).any():
#      #       print("[DEBUG] NaN trovato!")
#      #   print("lap_u range:", lap_u.min(), lap_u.max())
#      #   print("grad_alpha range:", grad_alpha.min(), grad_alpha.max())
#       #  print("dvdt range:", dvdt.min(), dvdt.max())

import numpy as np
def safe_exp(x, limit=100.0):
    x_clipped = np.clip(x, -limit, limit)
    return np.exp(x_clipped)

from core.grid import * 
from bssn.bssnstatevariables import NUM_BSSN_VARS
from bssn.tensoralgebra import EMTensor, get_bar_gamma_UU, get_bar_gamma_LL
from bssn.bssnstatevariables import idx_lapse


class ScalarMatter:
    """
    Massless scalar field for Choptuik critical collapse (spherical symmetry).
    Implements 3+1 Einstein–scalar without electromagnetic field.
    """

    def __init__(self, a_scalar_mu=0.0):
        # Pure collapse: massless scalar (μ = 0)
        self.scalar_mu       = 0.0
        self.NUM_MATTER_VARS = 2
        self.VARIABLE_NAMES  = ["u", "v"]
        self.PARITY          = np.array([1, 1])
        self.ASYMP_POWER     = np.array([0, 0])
        self.ASYMP_OFFSET    = np.array([0, 0])
        # Indici nel vettore di stato (dopo le variabili BSSN)
        self.idx_u    = NUM_BSSN_VARS
        self.idx_v    = NUM_BSSN_VARS + 1
        self.indices  = np.array([self.idx_u, self.idx_v])
        self.matter_vars_set = False
        self.u = []
        self.v = []
        self.du_dr = []
        self.d2u_dr2 = []
        
        

    def V_of_u(self, u):
        # Potenziale nullo
        return np.zeros_like(u)

    def dVdu(self, u):
        return np.zeros_like(u)

    def set_matter_vars(self, state_vector, bssn_vars, grid):
        # Estrae θ e Π dal vettore di stato
        self.u = state_vector[self.idx_u].copy()  # θ
        self.v = state_vector[self.idx_v].copy()  # Π

        # Prima derivata radiale ∂_r θ e Π
        d1 = grid.get_first_derivative(state_vector, [self.idx_u, self.idx_v])
        self.du_dr = np.ravel(d1[self.idx_u])
        self.dv_dr = np.ravel(d1[self.idx_v])

        # Seconda derivata radiale ∂_r² θ
        d2 = grid.get_second_derivative(state_vector, [self.idx_u])
        self.d2u_dr2 = np.ravel(d2[self.idx_u])

        # # === DEBUG CHECK: confronto con differenze finite manuali ===
        # N = grid.N
        # dr = grid.dr
        # i_test = N // 2   # punto circa a metà griglia
        # du_fd = (self.u[i_test+1] - self.u[i_test-1]) / (2.0 * dr)
        # d2u_fd = (self.u[i_test+1] - 2*self.u[i_test] + self.u[i_test-1]) / (dr**2)

        # if not hasattr(self, "_printed_test"):  # solo la prima volta
        #     print("[TEST] Punto", i_test)
        #     print("self.du_dr =", self.du_dr[i_test], " | FD approx =", du_fd)
        #     print("self.d2u_dr2 =", self.d2u_dr2[i_test], " | FD approx =", d2u_fd)
        #     self._printed_test = True

        self.matter_vars_set = True
    

    def get_emtensor(self, r, bssn_vars, background):
        assert self.matter_vars_set, "Call set_matter_vars() before get_emtensor()"
        N = r.size
        T = EMTensor(N)

        # Fattori metrici
        em4phi = safe_exp(-4.0 * bssn_vars.phi)
        bar_uu = get_bar_gamma_UU(r, bssn_vars.h_LL, background)
        bar_ll = get_bar_gamma_LL(r, bssn_vars.h_LL, background)

        grr_inv = bar_uu[:, i_x1, i_x1]

        inv16pi = 1.0 / (16.0 * np.pi)
        inv32pi = 1.0 / (32.0 * np.pi)

        grad2 = em4phi * grr_inv * (self.du_dr ** 2)

        # Componenti del tensore energia‐impulso
        T.rho = inv16pi * (self.v ** 2) + inv32pi * grad2

        Si = np.zeros((N, SPACEDIM))
        Si[:, i_x1] = inv16pi * self.v * self.du_dr
        T.Si = Si 

        Sij = np.zeros((N, SPACEDIM, SPACEDIM))
        Sij[:, i_x1, i_x1] = inv16pi * (self.du_dr ** 2)

        # for a in range(SPACEDIM):
        #     Sij[:, a, a] += -bar_ll[:, a, a] * inv32 * grad2
        
        for a in range(SPACEDIM):
            Sij[:, a, a] -= (safe_exp(4.0 * bssn_vars.phi) * bar_ll[:, a, a]) * inv32pi * grad2

        T.Sij = Sij

        T.S = em4phi * np.einsum('xjk,xjk->x', bar_uu, Sij)
        
        return T
    
    
    def compute_lap_u(self, r, grid, bssn_vars, background):
        phi    = bssn_vars.phi
        em6phi = safe_exp(-6.0 * phi)   # e^{-6φ}
        e2phi  = safe_exp( 2.0 * phi)

        bar_uu  = get_bar_gamma_UU(r, bssn_vars.h_LL, background)
        bar_grr = bar_uu[:, i_x1, i_x1]
    
        du = self.du_dr
        inner = (r**2) * e2phi * bar_grr * du

        d_inner = grid.derivs.drn_matrix[1] @ inner

        lap_u = em6phi * (1.0 / (r**2)) * d_inner
        lap_u[0] = 3.0 * safe_exp(-4.0 * phi[0]) * bar_grr[0] * self.d2u_dr2[0]
    
        return lap_u

        
    def get_matter_rhs(self, r, bssn_vars, bssn_d1, background, grid):
        assert self.matter_vars_set, "Call set_matter_vars() before get_matter_rhs()"

        dudt = bssn_vars.shift_U[:, i_x1] * self.du_dr - bssn_vars.lapse * self.v
        lap_u = self.compute_lap_u(r, grid, bssn_vars, background)

        alpha_grad_r = bssn_d1.lapse[:,i_x1]
        grad_alpha   = alpha_grad_r * self.du_dr
        grad_alpha[0] = 0.0
        
        em4phi = safe_exp(-4.0 * bssn_vars.phi)
        bar_uu = get_bar_gamma_UU(r, bssn_vars.h_LL, background)
        grr_inv  = bar_uu[:, i_x1, i_x1]
        
        dvdt = (
            bssn_vars.shift_U[:, i_x1] * self.dv_dr
            - bssn_vars.lapse * lap_u
            - em4phi * grr_inv * grad_alpha
            + bssn_vars.lapse * bssn_vars.K * self.v
        )
        
        return dudt, dvdt

        