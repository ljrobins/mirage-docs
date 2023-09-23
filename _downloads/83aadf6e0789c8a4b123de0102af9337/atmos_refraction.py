"""
Atmospheric Refraction
======================
Computing the effect of atmospheric refraction on observations
"""
import numpy as np
from scipy.optimize import fsolve

import mirage as mr
import mirage.vis as mrv


def apparent_refacted_elevation(pressure_mbar: np.ndarray, temp_kelvin: np.ndarray, el_true_rad: np.ndarray) -> np.ndarray:
    el_true_deg = np.rad2deg(el_true_rad)
    h_func = lambda hprime: \
        - hprime + pressure_mbar/temp_kelvin * (3.430289 * ((90 - hprime) - \
        mr.asind(0.9986047 * mr.sind(0.996714 * (90 - hprime)))) \
        - 0.01115929*(90-hprime)) + el_true_deg
    # If this function is zero, the correct apparent elevation has been identified 
    hprime = np.deg2rad(fsolve(h_func, el_true_deg))
    return hprime

true_el = np.linspace(0, np.pi/2, int(1e3))
apparent_el = apparent_refacted_elevation(1028.4463393, 277.594, true_el)

import matplotlib.pyplot as plt

plt.plot(np.rad2deg(true_el), np.rad2deg(apparent_el))
mrv.texit("Atmospheric Refraction", "True elevation [deg]", "Apparent elevation [deg]")
plt.ylim(0,90)
plt.xlim(0,90)
plt.show()