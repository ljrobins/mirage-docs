"""
Background Importance
=====================

Plotting the changes in background signal values over the course of one night
"""

import datetime

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr

# %%
# Defining a function we can use to plot various background signals


def hemisphere_signal(
    station: mr.Station,
    date: datetime.datetime,
    signal_kwargs: dict,
) -> None:
    true_signals = [k for k in signal_kwargs.keys() if signal_kwargs[k]]
    if len(true_signals) == len(signal_kwargs.keys()):
        signal_type = 'All Signals'
    else:
        signal_type = true_signals[0].capitalize()

    (g_az, g_el) = np.meshgrid(
        np.linspace(0, 2 * np.pi, 50),
        np.linspace(np.deg2rad(0), np.pi / 2, 50),
    )

    lde = []
    for date in dates:
        lde.append(station.az_el_to_eci(g_az.flatten(), g_el.flatten(), date))

    look_dirs_eci_eq_tiled = np.vstack(lde)
    dates_tiled = np.tile(dates, g_az.size)

    mr.tic(signal_type)
    sb = station.sky_brightness(dates_tiled, look_dirs_eci_eq_tiled, **signal_kwargs)
    mr.toc()

    print(f'Max:  {sb.max():.2e}')
    print(f'Mean: {sb.mean():.2e}')
    return sb


# %%
# Setting up observation conditions using an example Space Debris Telescope preset from Krag2003
# station = mr.Station(preset="lmt", lat_deg=33.776864, lon_deg=-84.363777) # Atlanta, GA
station = mr.Station(preset='pogs')
station.telescope.integration_time = 1.0
date = mr.utc(2023, 10, 1, 0, 0, 0)

# %%
# Plotting the background signal for scattered moonlight
signals = [
    'moonlight',
    'airglow',
    'integrated_starlight',
    'zodiac',
    'pollution',
    'twilight',
]
cs = ['k', 'm', 'c', 'b', 'r', 'g']

dates = mr.date_arange(date, date + mr.hours(14), mr.hours(0.5))
sun_ang_deg = np.rad2deg(mr.sun_angle_to_horizon(dates, station.itrf)).flatten()
is_astro_dark = sun_ang_deg < -18.0
is_nautical_dark = ~is_astro_dark & (sun_ang_deg < -12.0)
is_civil_dark = ~is_astro_dark & ~is_nautical_dark & (sun_ang_deg < -6.0)
is_light_dates = sun_ang_deg > 0.0

dates = dates[~is_light_dates]
hr_after_dark = mr.date_to_epsec(dates) / 3600

plt.figure(figsize=(8, 4))

for signal, c in zip(signals, cs):
    kwargs = {s: (False if s is not signal else True) for s in signals}
    sb = hemisphere_signal(station, dates, kwargs)
    sb = sb.reshape(-1, dates.size)
    label = signal.replace('_', ' ').capitalize()
    if len(label.split(' ')) > 1:
        label = label.split(' ')[1].capitalize()
    plt.subplot(1, 2, 1)
    plt.plot(hr_after_dark, np.max(sb, axis=0), c=c, marker='+', label=label)
    plt.xlabel('Hours after sunset')
    plt.ylabel('Signal rate maximum [ADU / pix / s]')
    plt.yscale('log')
    plt.grid(visible=True)
    plt.subplot(1, 2, 2)
    plt.plot(hr_after_dark, np.mean(sb, axis=0), c=c, marker='+', label=label)
    plt.xlabel('Hours after sunset')
    plt.ylabel('Signal rate mean [ADU / pix / s]')
    plt.yscale('log')
    plt.grid(visible=True)

plt.legend(loc='upper center', ncol=2, fancybox=True, shadow=True)

plt.tight_layout()
plt.show()
