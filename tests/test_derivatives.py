import numpy as np
import pytest
import scipy
import sympy
from sympy import abc

from source.derivatives import Derivatives
from source.spacing import *
from source.uservariables import NUM_GHOSTS


delta = np.finfo(float).eps

r_max = 8
min_dr = 1 / 32
max_dr = 1 / 16

abs_err = 1e-5
rel_err = 1e-2


class TestDerivatives:
    @classmethod
    def setup_class(cls):
        params = SinhSpacing.get_parameters(r_max, min_dr, max_dr)
        cls.sp = SinhSpacing(**params)
        print(cls.sp.dx)
        cls.der = Derivatives(cls.sp)
        cls.x = cls.sp.x
        cls.dnr_dxn = cls.sp._dnr_dxn()
        cls.r = cls.dnr_dxn[0]
        cls.dr_dx = cls.dnr_dxn[1]
        cls.dx = cls.der.dx
        cls.num_points = params["num_points"]

    def test_dxn(self):
        """Function to check that dxn_matrix is correct.

        Compute derivatives with this matrix on sinus function.
        """
        params = LinearSpacing.get_parameters(r_max, 1 / 32)
        sp = LinearSpacing(**params)
        der = Derivatives(sp)
        x = sp.x
        y_x = np.sin(x)
        num_points = y_x.size
        hn = sp.dx ** np.arange(7, dtype=np.uint8).reshape((-1, 1))
        dny_dxn = (der.dxn_matrix @ y_x) / hn

        expected = np.zeros((7, num_points))
        expected[[0, 4]] = np.sin(x)
        expected[[1, 5]] = np.cos(x)
        expected[[2, 6]] = -np.sin(x)
        expected[3] = -np.cos(x)
        # Expect 0 on ghost points.
        expected[:, :NUM_GHOSTS] = expected[:, -NUM_GHOSTS:] = 0

        assert dny_dxn == pytest.approx(expected, rel=rel_err)

    def test_drn(self):
        y_r = np.sin(self.r)
        num_points = self.num_points
        hn = (self.dx * self.dr_dx.reshape(1, -1)) ** np.arange(7, dtype=np.uint8).reshape((-1, 1))
        dny_drn = (self.der.drn_matrix @ y_r) / hn

        expected = np.zeros((7, num_points))
        expected[[0, 4]] = np.sin(self.r)
        expected[[1, 5]] = np.cos(self.r)
        expected[[2, 6]] = -np.sin(self.r)
        expected[3] = -np.cos(self.r)
        # Expect 0 on ghost points.
        expected[:, :NUM_GHOSTS] = expected[:, -NUM_GHOSTS:] = 0

        assert dny_drn == pytest.approx(expected, abs=rel_err)

    def test_advec_x(self):
        num_points = self.num_points
        y_x = np.sin(self.x)
        advec_y_x = (self.der.advec_x_matrix @ y_x) / self.dx

        expected = np.zeros((2, num_points))
        expected[:] = np.cos(self.x)
        # Expect 0 on ghost points.
        expected[0, :NUM_GHOSTS] = expected[1, -NUM_GHOSTS:] = 0

        assert advec_y_x == pytest.approx(expected, rel=rel_err)
