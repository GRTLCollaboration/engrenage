import numpy as np
import pytest
import scipy
import sympy
from sympy import abc

from source.derivatives import Derivatives
from source.spacing import SinhSpacing
from source.uservariables import NUM_GHOSTS


a = 1
delta = np.finfo(float).eps
num_points = 100
r_max = 5

# TODO: Check why error on dxn 6th derivative is weird
# TODO: For trucature error take sup and not current value


class TestDerivatives:
    @classmethod
    def setup_class(cls):
        cls.sp = SinhSpacing(num_points=num_points, r_max=r_max, a=a)
        cls.der = Derivatives(cls.sp)
        cls.x = cls.sp.x
        cls.dnr_dxn = cls.sp._dnr_dxn()
        cls.r = cls.dnr_dxn[0]
        cls.dr_dx = cls.dnr_dxn[1]
        cls.dx = cls.der.dx

    def test_dxn(self):
        """
        Function to check that dxn_matrix is correct.
        Compute derivatives with this matrix on exponential function.
        """
        y_x = np.exp(self.x)
        hn = self.dx ** np.arange(7, dtype=np.uint8).reshape((-1, 1))
        dny_dxn = (self.der.dxn_matrix @ y_x) / hn
        # Nth derivative of exp equals to exp so just repeat vector
        expected = y_x.reshape((1, -1)).repeat(7, axis=0)
        # Expect 0 on ghost points.
        expected[:, :NUM_GHOSTS] = expected[:, -NUM_GHOSTS:] = 0
        # dnf_dxn to compute truncate error
        dnf_dxn = y_x.reshape((1, -1)).repeat(9, axis=0)

        e_a = self.e_a(self.der.dxn_matrix, y_x, hn)
        e_t = self.e_t(self.der.dxn_matrix, dnf_dxn)
        abs_error = e_a + e_t
        rel_error = (dny_dxn - expected) / (abs_error + delta)

        assert rel_error == pytest.approx(np.zeros((7, num_points)), abs=1)

    def test_drn(self):
        y_r = np.exp(self.r)
        dr_dx = self.dnr_dxn[1]
        hn = (self.dx * self.dr_dx.reshape(1, -1)) ** np.arange(7, dtype=np.uint8).reshape((-1, 1))
        dny_drn = (self.der.drn_matrix @ y_r) / hn
        expected = y_r.reshape((1, num_points)).repeat(7, axis=0)
        # Expect 0 on ghost points.
        expected[:, :NUM_GHOSTS] = expected[:, -NUM_GHOSTS:] = 0
        e_a = self.e_a(self.der.drn_matrix, y_r, hn)

        # First compute derivatives of g = f o r function, here g = exp(a sinh x)
        # using sympy.
        # Order 8 derivative needed
        dnf_dxn = np.zeros((9, num_points))
        expr = sympy.exp(a * sympy.sinh(abc.x))
        for i in range(9):
            expr_fn = sympy.diff(expr, abc.x, i)
            dnf_dxn[i] = sympy.lambdify(abc.x, expr_fn, "numpy")(self.x)

        e_t = self.e_t(self.der.dxn_matrix, dnf_dxn) # / (self.h * dr_dx) ** np.arange(7, dtype=np.uint8).reshape((-1, 1))
        e = e_a + e_t
        assert (e >= 0).all(), "Computed error should be positive."
        rel_error = (dny_drn - expected) / (e + delta)
        assert rel_error == pytest.approx(np.zeros((7, num_points)), abs=2)

    def test_advec_x(self):
        y_x = np.exp(self.x)
        advec_y_x = (self.der.advec_x_matrix @ y_x) / self.dx
        # Nth derivative of exp equals to exp so just repeat vector
        expected = y_x.reshape((1, -1)).repeat(2, axis=0)
        # Expect 0 on ghost points.
        expected[0, :NUM_GHOSTS] = expected[1, -NUM_GHOSTS:] = 0

        e_a = delta * np.abs(self.der.advec_x_matrix) @ y_x / self.dx
        x = np.arange(num_points).reshape((-1, 1))
        power_of_n = scipy.spatial.distance_matrix(x, x).reshape(
            (1, num_points, -1)
        ).repeat(2, axis=0)
        factor = self.dx ** 3 / 24
        abs_advec_matrix = np.abs(self.der.advec_x_matrix) * power_of_n
        trunc_matrix = (
            np.triu(abs_advec_matrix)
            + np.tril(abs_advec_matrix).sum(axis=2, keepdims=True) * np.expand_dims(np.eye(num_points), axis=0)
        )
        e_t = factor * trunc_matrix @ np.abs(y_x)
        abs_error = e_a + e_t
        rel_error = (advec_y_x - expected) / (abs_error + delta)

        assert rel_error == pytest.approx(np.zeros((2, num_points)), abs=1)

    def e_a(self, matrix, vector, hn):
        # Estimate absolute error when computing derivative quotient.
        return delta * np.abs(matrix) @ np.abs(vector) / np.abs(hn)

    def e_t(self, matrix: np.ndarray, dnf_dxn: np.ndarray):
        order = np.array([0, 4, 4, 2, 2, 2, 2], dtype=np.uint8)
        hn = self.dx ** order
        der_order = np.arange(7, dtype=np.uint8) + order
        factor = (hn / scipy.special.factorial(der_order)).reshape((7, 1))
        factor[0] = 0
        x = np.arange(num_points).reshape((-1, 1))
        power_of_n = scipy.spatial.distance_matrix(x, x).reshape(
            (1, num_points, num_points)
        ) ** der_order.reshape((-1, 1, 1))
        abs_matrix = np.abs(matrix * power_of_n)
        trunc_matrix = (
                np.triu(abs_matrix)
                + np.tril(abs_matrix).sum(axis=2, keepdims=True) * np.expand_dims(np.eye(num_points), axis=0)
        )
        abs_dnf_dxn = np.abs(dnf_dxn[order])
        e_t = np.einsum("abc,ac->ab", trunc_matrix, abs_dnf_dxn)
        return factor * e_t
