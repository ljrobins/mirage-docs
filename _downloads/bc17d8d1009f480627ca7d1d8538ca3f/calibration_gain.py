"""
CCD Gain
========

Computing the gain of the POGS CCD from a set of twilight flats
"""

import os
from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize

import mirage as mr


def l1_fit(x, y):
    """
    Fits a line to data using L1 regression.
    """

    def cost_function(params, x, y):
        m, b = params
        return np.sum(np.abs(y - (m * x + b)))

    initial_guess = [1, 0]  # Initial guess for slope and intercept
    result = optimize.minimize(cost_function, initial_guess, args=(x, y))
    return result.x  # m, b for y = mx * b


def calibrate_sky_flats_to_gain(flats_dir: str) -> np.ndarray:
    # http://spiff.rit.edu/classes/phys445/lectures/gain/gain.html
    # Under "Measuring the gain -- a better technique"
    # as well as: https://www.mirametrics.com/tech_note_ccdgain.php
    flats_paths = [os.path.join(flats_dir, f) for f in os.listdir(flats_dir)]

    variances = []
    means = []
    flats_paths = sorted(
        flats_paths, key=lambda x: int(os.path.split(x)[1].split('.')[0])
    )

    for flat_path1, flat_path2 in pairwise(flats_paths):
        info1 = mr.info_from_fits(flat_path1, telescope=station.telescope, minimal=True)
        info2 = mr.info_from_fits(flat_path2, telescope=station.telescope, minimal=True)

        s1 = info1['ccd_adu'][*region] - 1010  # Subtracting the approximate bias level
        s2 = info2['ccd_adu'][*region] - 1010
        r = np.mean(s1) / np.mean(s2)
        s2 = s2 * r
        added = (s1 + s2) / 2
        subed = (s1 - s2) / 2

        mean_level = np.mean(added)
        rms = np.std(subed)
        variance = rms**2 / 2.0

        means.append(mean_level)
        variances.append(variance)

    means = np.array(means)
    variances = np.array(variances)
    m, b = l1_fit(means, variances)
    print(f'Gain: {1/m}')
    plt.scatter(means, variances, marker='+')
    plt.plot(means, means * m + b, label='best fit', color='k')
    plt.legend()
    plt.grid()
    plt.ylabel('Signal variance')
    plt.xlabel('Signal mean')
    plt.show()


def calibrate_sky_flats_to_k(flats_dir: str, g: float, rc: float) -> np.ndarray:
    # as well as: https://www.mirametrics.com/tech_note_ccdgain.php
    flats_paths = [os.path.join(flats_dir, f) for f in os.listdir(flats_dir)]

    ks = []
    means = []
    flats_paths = sorted(
        flats_paths, key=lambda x: int(os.path.split(x)[1].split('.')[0])
    )

    for flat_path in flats_paths:
        info = mr.info_from_fits(flat_path, telescope=station.telescope, minimal=True)
        s = info['ccd_adu'][*region]
        nc = np.std(s)
        sc = np.mean(s - 1010)
        k = np.sqrt((nc**2 - rc**2 - 1 / g * sc) / sc**2)
        ks.append(k)
        means.append(sc)

    ks = np.array(ks)
    means = np.array(means)

    m, b = l1_fit(means, ks)
    print('m, b')
    plt.scatter(means, ks, marker='+')
    plt.plot(means, means * m + b, label='best fit', color='k')
    plt.legend()
    plt.grid()
    plt.ylabel('Estimated Flat Field Variance $k$')
    plt.xlabel('Signal mean')
    plt.show()


region = (slice(430, 480), slice(370, 420))
station = mr.Station()
flats_dir = '/Users/liamrobinson/Downloads/2024_11_15/flat'
read_noise = 10.24  # ADU
gain = 5.6  # e-/ADU
# calibrate_sky_flats_to_gain(flats_dir)
calibrate_sky_flats_to_k(flats_dir, gain, read_noise)
