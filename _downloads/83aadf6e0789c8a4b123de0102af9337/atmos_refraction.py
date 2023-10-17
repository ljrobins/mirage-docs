"""
Atmospheric Refraction
======================
Computing the effect of atmospheric refraction on observations
"""
import numpy as np
from scipy.optimize import root

import mirage as mr
import mirage.vis as mrv


def apparent_refacted_elevation(
    pressure_mbar: np.ndarray, temp_kelvin: np.ndarray, el_true_rad: np.ndarray
) -> np.ndarray:
    el_true_deg = np.rad2deg(el_true_rad)
    h_func = (
        lambda hprime: -hprime
        + pressure_mbar
        / temp_kelvin
        * (
            3.430289
            * ((90 - hprime) - mr.asind(0.9986047 * mr.sind(0.996714 * (90 - hprime))))
            - 0.01115929 * (90 - hprime)
        )
        / 60
        + el_true_deg
    )
    # If this function is zero, the correct apparent elevation has been identified
    hprime = np.deg2rad(root(fun=h_func, x0=el_true_deg, method="diagbroyden").x)
    return hprime


true_el = np.deg2rad(np.linspace(0, 90, int(1e5)))

mr.tic()
apparent_el = apparent_refacted_elevation(1028.4463393, 277.594, true_el)
mr.toc()

import matplotlib.pyplot as plt

plt.plot(np.rad2deg(true_el), np.rad2deg(apparent_el - true_el))
mrv.texit(
    "Atmospheric Refraction", "True elevation $h$", "Refraction $R = h' - h$ [deg]"
)
plt.xlim(0, 90)
plt.ylim(0, 0.53)
plt.show()
