"""
This module tests that the derivatives computed in 'source.spacing' are correct using sympy.
"""


import numpy as np
import pytest
import sympy
from sympy.abc import x

from source.spacing import *
from source.uservariables import SpacingExtent


num_points: int = 200
r_max: float = 10


class BaseTestSpacing:
    """Abstract class to test that derivatives in a spacing are correct (with sympy)."""
    @classmethod
    def setup_class(cls):
        cls.r: sympy.core.Expr = ...
        cls.sp: Spacing = ...
        cls.dnr_dxn = cls.sp.dnr_dxn()

    def test_derivatives(self):
        expected = np.zeros((7, num_points))
        for i in range(7):
            f = sympy.lambdify(x, sympy.diff(self.r, x, i), "numpy")
            expected[i] = f(self.sp.x)

        assert self.dnr_dxn == pytest.approx(expected)


class TestLinearSpacing(BaseTestSpacing):
    """Test that derivatives in LinearSpacing are correct."""
    @classmethod
    def setup_class(cls):
        cls.r: sympy.core.Expr = x
        cls.sp = LinearSpacing(num_points, r_max, extent=SpacingExtent.FULL)
        cls.dnr_dxn = cls.sp._dnr_dxn()

    def test_get_parameters(self):
        min_dr = 1e-2
        params = LinearSpacing.get_parameters(r_max, min_dr=min_dr)
        sp = LinearSpacing(**params)
        dr = sp[0, 1:] - sp[0, :-1]
        assert dr.min() == pytest.approx(min_dr, rel=1e-1)


class TestSinhSpacing(BaseTestSpacing):
    """Test that derivatives in SinhSpacing are correct."""
    a = .5

    @classmethod
    def setup_class(cls):
        cls.r: sympy.core.Expr = cls.a * sympy.sinh(x)
        cls.sp = SinhSpacing(num_points, r_max, a=cls.a, extent=SpacingExtent.FULL)
        cls.dnr_dxn = cls.sp._dnr_dxn()

    def test_get_parameters(self):
        min_dr = 1e-2
        max_dr = 1e-1
        params = SinhSpacing.get_parameters(r_max, min_dr=min_dr, max_dr=max_dr, extent=SpacingExtent.FULL)
        sp = SinhSpacing(**params)
        dr = sp[0, 1:] - sp[0, :-1]
        assert dr.min() == pytest.approx(min_dr, rel=1e-1)
        assert dr.max() == pytest.approx(max_dr, rel=1e-1)


class TestCubicSpacing(BaseTestSpacing):
    """Test that derivatives in TanhSpacing are correct."""
    a = .5

    @classmethod
    def setup_class(cls):
        cls.r: sympy.core.Expr = cls.a * (x + x**3 / 3)
        cls.sp = CubicSpacing(num_points, r_max, a=cls.a, extent=SpacingExtent.FULL)
        cls.dnr_dxn = cls.sp._dnr_dxn()

    def test_get_parameters(self):
        min_dr = 1e-2
        max_dr = 1e-1
        params = CubicSpacing.get_parameters(r_max, min_dr=min_dr, max_dr=max_dr, extent=SpacingExtent.FULL)
        sp = CubicSpacing(**params)
        dr = sp[0, 1:] - sp[0, :-1]
        assert dr.min() == pytest.approx(min_dr, rel=1e-1)
        assert dr.max() == pytest.approx(max_dr, rel=1e-1)
