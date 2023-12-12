"""
Support Unobservability
=======================

This example shows how the support -- the final construction of the object -- is fundamentally unobservable from the light curve.
"""

import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

# %%
# Defining the nominal object

obj = mrv.SpaceObject("cube.obj")

v2 = obj.v.copy()
for fi, ni in zip(obj.f, obj.face_normals):
    for vind in fi:
        v2[vind] += ni

obj2 = mrv.SpaceObject(vertices_and_faces=(v2, obj.f))

pl = pv.Plotter()
mrv.render_spaceobject(pl, obj)
mrv.render_spaceobject(pl, obj2, color="gray")
pl.show()

# %%
# Calculating the irradiance difference for a change of 1 meter for one of the flat plates for an object in GEO

r_geo = 42164.0
r_plate = r_geo
r_plate2 = r_geo - 1e-3

irradiance_fraction_difference = 1 - (r_plate2 / r_plate) ** 2

print(
    f"The irradiance due to the closer plate is {irradiance_fraction_difference*100}% different"
)

# %%
# Figuring out the truncation losses in the CCD
obs = mr.Station(preset="pogs")
npix = obs.telescope.get_airy_disk_pixels()
ccd_signal = 1e4
trunc_variance = npix**2 / 24
trunc_std = np.sqrt(trunc_variance)

signal_difference = ccd_signal * irradiance_fraction_difference

print(f"The number of pixels in the airy disk is {npix}")
print(f"The standard deviation of the truncation noise is {trunc_std} ADU")
print(
    f"The difference in the signal due to plate distance is {signal_difference:.3e} ADU"
)
print(
    f"The truncation noise is {trunc_std / signal_difference} stronger than the signal difference"
)

SNR = signal_difference / np.sqrt(signal_difference + trunc_std)

print(f"The SNR is {SNR:.3e}")
