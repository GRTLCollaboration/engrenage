"""
This module provides functions to set a nice display on notebooks (no physics).
"""

import math

from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate._ivp.base import OdeSolver
from tqdm.auto import tqdm


# monkey patching the ode solvers with a progress bar
# save the old methods - we still need them
old_init = OdeSolver.__init__
old_step = OdeSolver.step


# define our own methods
def _new_init(self, fun, t0, y0, t_bound, vectorized, support_complex=False):
    # define the progress bar
    self.pbar = tqdm(
        total=t_bound - t0,
        unit="ut",
        initial=t0,
        ascii=True,
        desc="IVP",
    )
    self.last_t = t0

    # call the old method - we still want to do the old things too!
    old_init(self, fun, t0, y0, t_bound, vectorized, support_complex)


def _new_step(self):
    # call the old method
    old_step(self)

    # update the bar
    tst = self.t - self.last_t
    self.pbar.update(tst)
    self.last_t = self.t

    # close the bar if the end is reached
    if self.t >= self.t_bound:
        self.pbar.close()


def update_ode_solver():
    """Update the scipy OdeSolver to add a nice progress bar during solving."""

    OdeSolver.__init__ = _new_init
    OdeSolver.step = _new_step


def set_grid_on_ax(
        ax: plt.Axes,
        r: np.ndarray,
        r_max: float = None,
        display_number: int = 64
):
    """Set a grid adapted to the r coordinate on a pyplot ax."""

    if r_max is None:
        r_max = round(r[-1])
    if np.abs(r[0] + r[-1]) <= 1e-8:
        ax.set_xticks(np.linspace(-r_max, r_max, 9))
        eps = 2 * r_max / 20
        ax.set_xlim(-r_max - eps, r_max + eps)
    else:
        ax.set_xticks(np.linspace(0, r_max, 9))
        eps = r_max / 20
        ax.set_xlim(-eps, r_max + eps)
    r_plot = list(x for x in r if np.abs(x) < r_max)
    n = math.ceil(len(r_plot) / display_number)
    ax.set_xticks(r_plot[::n], minor=True)
    ax.set_xlabel("$r$")

    ax.grid(which="major", alpha=0.8)
    ax.grid(which="minor", alpha=0.2)
