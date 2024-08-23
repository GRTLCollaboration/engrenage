import numpy as np
import pytest

from source.grid import Grid
from source.spacing import LinearSpacing
from source.uservariables import *


r_max = 8
min_dr = 1 / 128

abs_err = 1e-5


def f(x):
    return np.exp(-x**2)


def df_dx(x):
    return -2 * x * np.exp(-x**2)


def d2f_dx2(x):
    return -2 * np.exp(-x**2) * (1 - 2 * x**2)


class TestHalfGrid:
    @classmethod
    def setup_class(cls):
        params = LinearSpacing.get_parameters(r_max, min_dr)
        sp = LinearSpacing(**params)
        cls.grid = Grid(sp)

    def test_inner_boundary_fill(self):
        r = self.grid.r
        state = np.where(np.expand_dims(PARITY, axis=1) == -1, r, r**2)
        expected = state.copy()
        self.grid.fill_inner_boundary(state)
        assert state == pytest.approx(expected)

    def test_outer_boundary_fill(self):
        state = (
                (np.arange(NUM_VARS) + 1).reshape((-1, 1)) * np.power(self.grid.r, ASYMP_POWER.reshape((-1, 1)))
                + ASYMP_OFFSET.reshape((-1, 1))
        )
        expected = state.copy()
        self.grid.fill_outer_boundary(state)
        assert state == pytest.approx(expected)

    def test_first_derivative(self):
        num_points = self.grid.num_points
        state = np.zeros((NUM_VARS, num_points))
        state[1] = f(self.grid.r)
        dstate_dr = self.grid.get_first_derivative(state, [1])
        expected = df_dx(self.grid.r)
        expected[:NUM_GHOSTS] = expected[-NUM_GHOSTS:] = 0
        assert np.abs(dstate_dr[1] - expected) == pytest.approx(np.zeros(num_points), abs=abs_err)

    def test_advection(self):
        num_points = self.grid.num_points
        state = np.zeros((NUM_VARS, num_points))
        state[2] = f(self.grid.r)
        direction = np.where(self.grid.r < self.grid.r[num_points // 2], 1, 0)
        dstate_dr = self.grid.get_advection(state, direction, [2])
        expected = df_dx(self.grid.r)
        assert np.abs(dstate_dr[2] - expected) == pytest.approx(np.zeros(num_points), abs=abs_err)

    def test_second_derivative(self):
        num_points = self.grid.num_points
        state = np.zeros((NUM_VARS, num_points))
        state[3] = f(self.grid.r)
        d2state_dr2 = self.grid.get_second_derivative(state, [3])
        expected = d2f_dx2(self.grid.r)
        expected[:NUM_GHOSTS] = expected[-NUM_GHOSTS:] = 0
        assert np.abs(d2state_dr2[3] - expected) == pytest.approx(np.zeros(num_points), abs=abs_err)


class TestFullGrid:
    @classmethod
    def setup_class(cls):
        params = LinearSpacing.get_parameters(r_max, min_dr, extent=SpacingExtent.FULL)
        sp = LinearSpacing(**params)
        cls.grid = Grid(sp)

    def test_first_derivative(self):
        num_points = self.grid.num_points
        state = np.zeros((NUM_VARS, num_points))
        state[1] = f(self.grid.r)
        dstate_dr = self.grid.get_first_derivative(state, [1])
        expected = df_dx(self.grid.r)
        expected[:NUM_GHOSTS] = expected[-NUM_GHOSTS:]
        assert np.abs(dstate_dr[1] - expected) == pytest.approx(np.zeros(num_points), abs=abs_err)

    def test_advection(self):
        num_points = self.grid.num_points
        state = np.zeros((NUM_VARS, num_points))
        state[2] = f(self.grid.r)
        direction = np.where(self.grid.r < self.grid.r[num_points // 2], 1, 0)
        dstate_dr = self.grid.get_advection(state, direction, [2])
        expected = df_dx(self.grid.r)
        assert np.abs(dstate_dr[2] - expected) == pytest.approx(np.zeros(num_points), abs=abs_err)

    def test_second_derivative(self):
        num_points = self.grid.num_points
        state = np.zeros((NUM_VARS, num_points))
        state[3] = f(self.grid.r)
        d2state_dr2 = self.grid.get_second_derivative(state, [3])
        expected = d2f_dx2(self.grid.r)
        expected[:NUM_GHOSTS] = expected[-NUM_GHOSTS:]
        assert np.abs(d2state_dr2[3] - expected) == pytest.approx(np.zeros(num_points), abs=abs_err)

    def test_inner_boundary_fill(self):
        state = (
                (np.arange(NUM_VARS) + 1).reshape((-1, 1)) * np.power(self.grid.r, ASYMP_POWER.reshape((-1, 1)))
                + ASYMP_OFFSET.reshape((-1, 1))
        )
        expected = state.copy()
        self.grid.fill_inner_boundary(state)
        assert state == pytest.approx(expected)

    def test_outer_boundary_fill(self):
        state = (
                (np.arange(NUM_VARS) + 1).reshape((-1, 1)) * np.power(self.grid.r, ASYMP_POWER.reshape((-1, 1)))
                + ASYMP_OFFSET.reshape((-1, 1))
        )
        expected = state.copy()
        self.grid.fill_outer_boundary(state)
        assert state == pytest.approx(expected)
