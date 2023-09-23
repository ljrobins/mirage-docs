"""
CCD Rendering
=============

Renders a synthetic CCD image of an observation taken by the POGS telescope
"""

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr
import mirage.vis as mrv

station = mr.Station("pogs")
station.telescope.fwhm = 4

date = mr.utc(2023, 1, 1, 7)
deltat = mr.seconds(10)
dates = np.array([date, date + deltat])

obj = mr.SpaceObject("matlib_hylas4.obj", identifier=26853)
r_obj_eci = obj.propagate(dates)

sv = mr.sun(dates)
nadir = -mr.hat(r_obj_eci)
attitude = mr.AlignedAndConstrainedAttitude(
    v_align=nadir,
    v_const=sv,
    dates=dates,
    axis_order=(1, 2, 0),
)
obj_lc_sampler, _ = station.observe_light_curve(
    obj,
    attitude,
    mr.Brdf("phong"),
    dates,
    use_engine=True,
    instance_count=1,
    model_scale_factor=10,
    rotate_panels=True,
)
lc_adu = obj_lc_sampler()

adu_grid_streaked_sampled = station.telescope.ccd.generate_ccd_image(dates, station, r_obj_eci, lc_adu)
plt.imshow(np.log10(adu_grid_streaked_sampled), cmap="gray")
mrv.texit(
    f'CCD Image of {obj.satnum} at {dates[0].strftime("%Y-%m-%d %H:%M:%S")} UTC', "", ""
)
plt.grid(False)
plt.show()
