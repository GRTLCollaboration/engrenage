"""
This module defines different spacings for the r coordinate.

A spacing is a function which defines where the physical points r should be.
Some examples of how to use spacings in the code may be found in the notebook examples/spacing.py.
New spacings should inherit from Spacing abstract base class.
"""

from abc import ABC, abstractmethod
from math import ceil
from enum import Enum

import numpy as np
from scipy.optimize import root_scalar

__all__ = [
    "NUM_GHOSTS",
    "SpacingExtent",
    "Spacing",
    "LinearSpacing",
    "SinhSpacing",
    "CubicSpacing",
]

NUM_GHOSTS: int = 3
    
class SpacingExtent(Enum):
    HALF = 0
    FULL = 1

class Spacing(ABC):
    def __init__(self, num_points: int, r_max: float, extent: SpacingExtent, **kwargs):
        # num_points must be an even number so that r is never equal to zero.
        # Full extent means the spacing ranges from -r_max to r_max
        # and half extent means it ranges from 0 to r_max.
        self.r_max = r_max
        self.extent = extent
        x_max = self.x_of_r(r_max, **kwargs)

        if extent == SpacingExtent.FULL:
            if num_points % 2 == 1:
                raise ValueError("Number of points must be even for full extent.")
            self.x, self.dx = np.linspace(-x_max, x_max, num_points, retstep=True)
        else:
            all_points = 2 * (num_points - NUM_GHOSTS)
            self.x, self.dx = np.linspace(-x_max, x_max, all_points, retstep=True)
            self.x = self.x[-num_points:]

        self.dnr_dxn = self._dnr_dxn()
        self.min_dr = np.min(self.dnr_dxn[0, 1:] - self.dnr_dxn[0, :-1])

    def __getitem__(self, item):
        return self.dnr_dxn[item]

    @classmethod
    @abstractmethod
    def x_of_r(cls, r: float, **kwargs):
        """Return the inverse of the spacing function, i.e. x as a function of r."""
        pass

    @abstractmethod
    def _dnr_dxn(self) -> np.ndarray:
        """Return the spacing function and its derivatives in an array.

        The returned array dnr_dxn should be of shape [7, N] (N the number of points),
        such that dnr_dxn[i] is the i-th derivative of r with respect to x.
        (dnr_dxn[0] = r(x), dnr_dxn[1] = r'(x), etc)
        """
        pass

    @staticmethod
    @abstractmethod
    def get_parameters(*args, **kwargs) -> dict:
        """Return the parameters to use to achieve a specified min_dr and max_dr in the spacing.

        The parameters should be return as a dict and may be used as keyword arguments.
        See the notebook in examples/spacing.py for some examples.
        """
        pass


class LinearSpacing(Spacing):
    """Class for spacing r = x."""

    def __init__(
        self, num_points: int, r_max: float, extent: SpacingExtent = SpacingExtent.HALF
    ):
        super().__init__(num_points, r_max, extent)

    @classmethod
    def x_of_r(cls, r: float, **kwargs):
        return r

    def _dnr_dxn(self):
        x = self.x
        dnr_dxn = np.zeros((7, x.size))
        dnr_dxn[0] = x
        dnr_dxn[1] = np.ones_like(x)
        return dnr_dxn

    @staticmethod
    def get_parameters(
            r_max: float,
            min_dr: float,
            extent: SpacingExtent = SpacingExtent.HALF,
    ):
        parameters = {"r_max": r_max, "extent": extent}
        x_max = r_max
        dx = min_dr

        if extent == SpacingExtent.FULL:
            num_points = ceil(2 * x_max / dx + 1)
            num_points += num_points % 2
        else:
            num_points = ceil(x_max / dx + NUM_GHOSTS + 1 / 2)

        parameters["num_points"] = num_points
        return parameters


class SinhSpacing(Spacing):
    """Class for spacing r = a sinh(x)."""

    def __init__(
        self,
        num_points: int,
        r_max: float,
        extent: SpacingExtent = SpacingExtent.HALF,
        a: float = 1,
    ):
        self.a = a
        super().__init__(num_points, r_max, extent, a=a)

    @classmethod
    def x_of_r(cls, r: float, a: float = 1):
        return np.arcsinh(r / a)

    def _dnr_dxn(self):
        a, x = self.a, self.x
        dnr_dxn = np.zeros((7, x.size))
        dnr_dxn[[0, 2, 4, 6]] = a * np.sinh(x)
        dnr_dxn[[1, 3, 5]] = a * np.cosh(x)
        return dnr_dxn

    @staticmethod
    def get_parameters(
            r_max: float,
            min_dr: float,
            max_dr: float,
            extent: SpacingExtent = SpacingExtent.HALF,
    ):
        if min_dr >= max_dr:
            raise ValueError("This function only works with min_dr << max_dr.")

        parameters = {"r_max": r_max, "extent": extent}

        x_max = root_scalar(
            lambda x: np.tanh(x) - np.sinh(x) * min_dr / max_dr,
            bracket=[min_dr / max_dr, max_dr / min_dr]
        ).root
        a = parameters["a"] = r_max / np.sinh(x_max)
        dx = min_dr / a

        if extent == SpacingExtent.FULL:
            num_points = ceil(2 * x_max / dx + 1)
            num_points += num_points % 2
        else:
            num_points = ceil(x_max / dx + NUM_GHOSTS + 1 / 2)

        parameters["num_points"] = num_points
        return parameters


class CubicSpacing(Spacing):
    """Class for spacing r = a (x + x^3 / 3)."""

    def __init__(
        self,
        num_points: int,
        r_max: float,
        extent: SpacingExtent = SpacingExtent.HALF,
        a: float = 1,
    ):
        self.a = a
        super().__init__(num_points, r_max, extent, a=a)

    @classmethod
    def x_of_r(cls, r: float, a: float = 1):
        return root_scalar(
            lambda x: a * (x + x**3 / 3) - r, bracket=[0, r / a]
        ).root

    def _dnr_dxn(self):
        a, x = self.a, self.x
        dnr_dxn = np.zeros((7, x.size))
        dnr_dxn[0] = a * (x + x**3 / 3)
        dnr_dxn[1] = a * (1 + x**2)
        dnr_dxn[2] = 2 * a * x
        dnr_dxn[3] = 2 * a
        return dnr_dxn

    @staticmethod
    def get_parameters(
            r_max: float,
            min_dr: float,
            max_dr: float,
            extent: SpacingExtent = SpacingExtent.HALF,
    ):
        if min_dr >= max_dr:
            raise ValueError("This function only works with min_dr << max_dr.")

        parameters = {"r_max": r_max, "extent": extent}

        x_max = np.sqrt(max_dr / min_dr - 1)
        a = parameters["a"] = r_max / (x_max + x_max**3 / 3)
        dx = min_dr / a

        if extent == SpacingExtent.FULL:
            num_points = ceil(2 * x_max / dx + 1)
            num_points += num_points % 2
        else:
            num_points = ceil(x_max / dx + NUM_GHOSTS + 1 / 2)

        parameters["num_points"] = num_points
        return parameters
