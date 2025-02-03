import numpy as np

from core.spacing import Spacing, NUM_GHOSTS

class Derivatives:
    """
    Represents the derivatives on a grid defines by the points in spacing.
    It uses finite difference method: the code provide derivation matrices up to order 6.
    They may be used to compute derivatives of functions using dot product.
    On the ghost points, (3 assumed by default), the derivatives are not computed.
    """
    def __init__(
        self,
        spacing: Spacing,
    ):

        self.dnr_dxn = spacing.dnr_dxn
        self.dx = spacing.dx
        self.num_points = n = self.dnr_dxn.shape[1]

        # Create a matrix with stencils for derivative from order 0 to order 6
        self.dxn_matrix = np.zeros((7, n, n))
        self.drn_matrix = np.zeros((7, n, n))
        self.compute_dxn_matrix()
        self.compute_drn_matrix()

        # Compute left and right advection matrices
        self.advec_x_matrix = np.zeros((2, n, n))
        self.compute_advec_x_matrix()

    def compute_dxn_matrix(self):
        """
        Fill the derivation matrices for x coordinate, i.e. evenly spaced points.
        The spacing is assumed to be one, scaling is done afterward.
        The finite difference coefficients may be found on
        https://en.wikipedia.org/wiki/Finite_difference_coefficient.
        """
        # For readability
        n = self.num_points
        # 0. Oth order derivative is identity (not used but for completeness).
        self.dxn_matrix[0] = np.eye(n)
        # 1. Create fourth order first derivation matrix.
        # Stencil [1/12, -2/3, 0, 2/3, -1/12]
        self.dxn_matrix[1] = (
            1 / 12 * np.eye(n, k=-2)
            - 2 / 3 * np.eye(n, k=-1)
            + 2 / 3 * np.eye(n, k=1)
            - 1 / 12 * np.eye(n, k=2)
        )
        # 2. Create fourth order second derivation matrix.
        # Stencil [-1/12, 4/3, -5/2, 4/3, -1/12]
        self.dxn_matrix[2] = (
            -1 / 12 * np.eye(n, k=-2)
            + 4 / 3 * np.eye(n, k=-1)
            - 5 / 2 * np.eye(n, k=-0)
            + 4 / 3 * np.eye(n, k=1)
            - 1 / 12 * np.eye(n, k=2)
        )
        # 3. Create second order third derivation matrix.
        # Stencil [-1/2, 1, 0, -1, 1/2]
        self.dxn_matrix[3] = (
            -1 / 2 * np.eye(n, k=-2)
            + 1 * np.eye(n, k=-1)
            - 1 * np.eye(n, k=1)
            + 1 / 2 * np.eye(n, k=2)
        )
        # 4. Create second order fourth derivation matrix.
        # Stencil: [-1/2, 1, 0, -1, 1/2]
        self.dxn_matrix[4] = (
            1 * np.eye(n, k=-2)
            - 4 * np.eye(n, k=-1)
            + 6 * np.eye(n, k=0)
            - 4 * np.eye(n, k=1)
            + 1 * np.eye(n, k=2)
        )
        # 5. Create second order fifth derivation matrix.
        # Stencil: [-1, 2, -5/2, 0, 5/2, -2, 1]
        self.dxn_matrix[5] = (
            -1 / 2 * np.eye(n, k=-3)
            + 2 * np.eye(n, k=-2)
            - 5 / 2 * np.eye(n, k=-1)
            + 5 / 2 * np.eye(n, k=1)
            - 2 * np.eye(n, k=2)
            + 1 / 2 * np.eye(n, k=3)
        )
        # 6. Create second order sixth derivation matrix.
        # Stencil: [1, -6, 15, -20, 15, -6, 1]
        self.dxn_matrix[6] = (
            1 * np.eye(n, k=-3)
            - 6 * np.eye(n, k=-2)
            + 15 * np.eye(n, k=-1)
            - 20 * np.eye(n, k=0)
            + 15 * np.eye(n, k=1)
            - 6 * np.eye(n, k=2)
            + 1 * np.eye(n, k=3)
        )

        # Do not compute derivatives for ghost points.
        self.dxn_matrix[:, :NUM_GHOSTS] = self.dxn_matrix[:, -NUM_GHOSTS:] = 0

    def compute_drn_matrix(self):
        # Quantities need to be scaled for real derivatives (by a factor of 1 / (dr_dx * h)**n for nth derivative)
        dr_dx = self.dnr_dxn[1]
        self.drn_matrix[0] = self.dxn_matrix[0]
        self.drn_matrix[1] = self.dxn_matrix[1]
        self.drn_matrix[2] = (
                self.dxn_matrix[2]
                - self.dx * np.diag(self.dnr_dxn[2] / dr_dx) @ self.drn_matrix[1]
        )
        self.drn_matrix[3] = (
                self.dxn_matrix[3]
                - self.dx * np.diag(3 * self.dnr_dxn[2] / dr_dx) @ self.drn_matrix[2]
                - self.dx ** 2 * np.diag(self.dnr_dxn[3] / dr_dx) @ self.drn_matrix[1]
        )
        self.drn_matrix[4] = (
                self.dxn_matrix[4]
                - self.dx * np.diag(6 * self.dnr_dxn[2] / dr_dx) @ self.drn_matrix[3]
                - self.dx ** 2 * np.diag(4 * self.dnr_dxn[3] / dr_dx + 3 * (self.dnr_dxn[2] / dr_dx) ** 2) @ self.drn_matrix[2]
                - self.dx ** 3 * np.diag(self.dnr_dxn[4] / dr_dx) @ self.drn_matrix[1]
        )
        self.drn_matrix[5] = (
                self.dxn_matrix[5]
                - self.dx * np.diag(10 * self.dnr_dxn[2] / dr_dx) @ self.drn_matrix[4]
                - self.dx ** 2 * np.diag(10 * self.dnr_dxn[3] / dr_dx + 15 * (self.dnr_dxn[2] / dr_dx) ** 2) @ self.drn_matrix[3]
                - self.dx ** 3 * np.diag(10 * self.dnr_dxn[3] * self.dnr_dxn[2] / dr_dx ** 2 + 5 * self.dnr_dxn[4] / dr_dx) @ self.drn_matrix[2]
                - self.dx ** 4 * np.diag(self.dnr_dxn[5] / dr_dx) @ self.drn_matrix[1]
        )
        self.drn_matrix[6] = (
                self.dxn_matrix[6]
                - self.dx * np.diag(15 * self.dnr_dxn[2] / dr_dx) @ self.drn_matrix[5]
                - self.dx ** 2 * np.diag(45 * (self.dnr_dxn[2] / dr_dx) ** 2 + 20 * self.dnr_dxn[3] / dr_dx) @ self.drn_matrix[4]
                - self.dx ** 3 * np.diag(15 * self.dnr_dxn[4] / dr_dx + 60 * self.dnr_dxn[3] * self.dnr_dxn[2] / dr_dx ** 2 + 15 * (self.dnr_dxn[2] / dr_dx) ** 3) @ self.drn_matrix[3]
                - self.dx ** 4 * np.diag(6 * self.dnr_dxn[5] / dr_dx + 15 * self.dnr_dxn[4] * self.dnr_dxn[2] / dr_dx ** 2 + 10 * (self.dnr_dxn[3] / dr_dx) ** 2) @ self.drn_matrix[2]
                - self.dx ** 5 * np.diag(self.dnr_dxn[6] / dr_dx) @ self.drn_matrix[1]
        )

    def compute_advec_x_matrix(self):
        # Left advection.
        # Third order backward stencil: [-1/3, 3/2, -3, 11/6]
        self.advec_x_matrix[0] = (
                -1 / 3 * np.eye(self.num_points, k=-3)
                + 3 / 2 * np.eye(self.num_points, k=-2)
                - 3 * np.eye(self.num_points, k=-1)
                + 11 / 6 * np.eye(self.num_points, k=0)
        )
        # Right advection.
        # Third order forward stencil: [-11/6, 3, -3/2, 1/3]
        self.advec_x_matrix[1] = (
            -11 / 6 * np.eye(self.num_points, k=0)
            + 3 * np.eye(self.num_points, k=1)
            - 3 / 2 * np.eye(self.num_points, k=2)
            + 1 / 3 * np.eye(self.num_points, k=3)
        )

        # Do not compute advection where it is not possible because of boundaries.
        self.advec_x_matrix[0, :NUM_GHOSTS] = 0
        self.advec_x_matrix[1, -NUM_GHOSTS:] = 0
