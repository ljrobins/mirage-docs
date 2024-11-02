"""
Shadow Functions
================

Comparing the effect of the atmosphere on penumbra shadow modeling
"""

import datetime

import matplotlib.pyplot as plt
import shado

import mirage as mr
import mirage.vis as mrv


def plot_shadow_function_for_dates(obj, dates) -> None:
    plt.figure(figsize=(8, 4))
    r = obj.propagate(dates)
    s = mr.sun(dates)
    sf = shado.shadow_function(r, s)
    sf_mine = mr.sun_irradiance_fraction(dates, r)

    plt.plot(dates, sf, label='Liu et al. 2019')
    plt.plot(dates, sf_mine, label='Krag 2003')
    mrv.texit(
        f'{obj.sat.name}',
        f"Date UTC ({dates[0].strftime('%b %d %Y')})",
        'Irradiance fraction',
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


# %%
# Galileo (MEO)

galileo = mr.SpaceObject('cube.obj', identifier='2011-060A')
d0 = mr.utc(2015, 1, 11, 18, 33, 34)
galileo_dates = mr.date_linspace(d0 - mr.minutes(1), d0 + mr.minutes(1), 100)
plot_shadow_function_for_dates(galileo, galileo_dates)

# %%
# GRACE (LEO)

grace = mr.SpaceObject('cube.obj', identifier='GRACE1')
d0 = mr.utc(2007, 1, 20, 1, 27, 25)
grace_dates = mr.date_linspace(d0 - mr.seconds(20), d0 + mr.seconds(20), 100)
plot_shadow_function_for_dates(grace, grace_dates)

# %%
# SL-3 R/B (LEO)

sl3rb = mr.SpaceObject('cube.obj', identifier='1973-015B')
d0, df = (
    datetime.datetime.fromisoformat('2021-05-03 22:35:06.904337+02:00'),
    datetime.datetime.fromisoformat('2021-05-03 22:36:38.541901+02:00'),
)
sl3rb_dates = mr.date_linspace(d0, df, 100)
plot_shadow_function_for_dates(sl3rb, sl3rb_dates)
