"""
Obsevations for Inversion
=========================

Plotting the observations used for inversion, along with their periodograms
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from astropy.timeseries import LombScargle

import mirage as mr

min_frequency = 6 / 360  # Hz
max_frequency = 50 / 360  # Hz
spp = 100
rev = '10'

for case in ['bw', 'rb', 'dmsp']:
    # if case == 'dmsp':
    #     continue
    real = False
    if case == 'dmsp':
        case_name = 'DMSP'
        real = True
    elif case == 'rb':
        case_name = 'Rocket Body'
    elif case == 'bw':
        case_name = 'Box-Wing'

    df = pl.read_parquet(
        os.path.join(
            os.environ['SRCDIR'],
            f'../scratch/asc26/data/r{rev}/df_{case}_cal_f.parquet',
        )
    )
    df_peaks = pl.read_parquet(
        os.path.join(
            os.environ['SRCDIR'],
            f'../scratch/asc26/data/r{rev}/sdlr_peaks_{case}.parquet',
        )
    )
    epsecs = df['epsecs'].to_numpy()

    peaks = df_peaks['peak'].to_numpy()
    troughs = peaks + df_peaks['dd'].to_numpy()
    peak_inds = df_peaks['peak_ind'].to_numpy()
    trough_inds = peak_inds + df_peaks['dind'].to_numpy()
    node_depths = np.concatenate((peaks, troughs))
    node_inds = np.concatenate((peak_inds, trough_inds))
    node_depths = node_depths[np.argsort(node_inds).flatten()]
    node_inds = np.sort(node_inds)
    node_epsecs = epsecs[node_inds]

    print(df.columns)
    if case != 'dmsp':
        r_obs = np.array(df['r_station_j2000'].to_list())
        r_obj = np.array(df['r_obj_j2000'].to_list())
        ovi = mr.hat(r_obs - r_obj)
    else:
        ovi = np.array(df['ovi'].to_list())

    ang_rate_deg = mr.angle_between_vecs(
        ovi[:-1], ovi[1:], deg=True
    ).flatten() / np.diff(epsecs)
    max_ang_rate = ang_rate_deg.max()

    # print(ang_rate_deg)
    # plt.plot(epsecs[1:], ang_rate_deg)
    # plt.show()
    # endd

    lc = df['lc'].to_numpy()
    depth = df['est_depth_poly'].to_numpy()
    frequency_d, power_d = LombScargle(epsecs, depth).autopower(
        minimum_frequency=min_frequency,
        maximum_frequency=max_frequency,
        samples_per_peak=spp,
    )
    frequency_l, power_l = LombScargle(epsecs, lc).autopower(
        minimum_frequency=min_frequency,
        maximum_frequency=max_frequency,
        samples_per_peak=spp,
    )

    plt.figure(figsize=(8, 7))
    plt.subplot(3, 1, 1)
    plt.scatter(epsecs, depth, s=1, alpha=0.3, label='Range residuals')

    l3 = plt.scatter(
        node_epsecs, node_depths, c='r', alpha=1.0, label='Leading edge extrema'
    )

    plt.plot(np.nan, np.nan)
    plt.plot(epsecs, lc / lc.mean(), label='Light curve')
    plt.ylabel('Residual range [m] / flux [nondim]')
    plt.legend()
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.xlabel('Synodic rate [deg/s]')
    plt.ylabel('Lomb-Scargle Power')
    plt.suptitle(f'{case_name} Observations')
    plt.xscale('log')
    plt.grid(True, which='both', ls='--', lw=0.5)

    for f, p, name in (
        (frequency_d, power_d, 'SDLR'),
        (frequency_l, power_l, 'light curve'),
    ):
        peak_idx = p.argmax()
        peak_rot_rate = f[peak_idx] * 360  # convert to deg/s

        plt.plot(f * 360, p, label=f'{name} periodogram', lw=2)
        plt.axvline(
            peak_rot_rate,
            color='red',
            linestyle='--',
            label=f'{name.capitalize()} peak: {peak_rot_rate:.2f} deg/s',
        )
    plt.tight_layout()
    old_lims = plt.ylim()
    plt.ylim(max(old_lims[0], np.percentile(p, 2)), old_lims[1] + 0.1)
    plt.yscale('log')
    plt.legend()

    plt.subplot(3, 1, 3)

    multiples = [1, 2, 3, 4]
    n = int(1e5)

    # target rotating at omega deg/s
    # progressing w.r.t. observer at Omega deg/s
    # after 360/omega seconds, object has rotated 360 deg sidereally
    # after 360/omega seconds, object has rotated 360 +/- 360 * Omega/omega deg, which we observe in the signal
    # backing out, omega_app = (360 +/- 360 * Omega/omega) / (360/omega) = (1 +/- Omega/omega) * omega = omega +/- Omega
    # if Omega is the max angular rate of the object w.r.t. the observer for the whole pass, this is nice and conservative ^

    data = []
    samples = []
    for m in multiples:
        for f, p, _name in (
            (frequency_d, power_d, 'SDLR'),
            (frequency_l, power_l, 'light curve'),
        ):
            fmax = f[p.argmax()] * 360 / m
            samples.append(
                np.random.uniform(
                    low=fmax - max_ang_rate, high=fmax + max_ang_rate, size=n
                )
            )
            data.append(dict(min=fmax - max_ang_rate, max=fmax + max_ang_rate))
    df_omega = pl.DataFrame(data)

    print(df_omega)
    samples = np.concatenate(samples)
    plt.hist(samples, bins=1000, density=True)

    if not real:
        plt.axvline(
            180 / np.pi * np.linalg.norm(df['w_of_t'][0].to_numpy()),
            color='red',
            linestyle='--',
            label='True sidereal rotation rate',
        )
    plt.xlabel('Rotation rate [deg/s]')
    plt.ylabel(r'$f_\omega(\omega)$ sampling density')
    plt.grid()
    plt.tight_layout()
    if case != 'dmsp':
        plt.legend()
    plt.show()

    np.save(
        os.path.join(os.environ['SRCDIR'], f'../scratch/asc26/data/w_{case}'),
        np.deg2rad(samples[np.random.randint(0, samples.size, samples.size)])[
            : int(1e5)
        ],
    )
