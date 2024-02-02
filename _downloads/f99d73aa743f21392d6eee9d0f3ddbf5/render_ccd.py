"""
CCD Rendering
=============

Renders a synthetic CCD image of an observation taken by the POGS telescope
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

# %%
# Loading a fits image from the Purdue Optical Ground Station

# ccd_dir = os.path.join(os.environ["SRCDIR"], "..", "data")
# fits_path = os.path.join(ccd_dir, "00130398.fit") # 3 in belt
# fits_path = os.path.join(
#     os.environ["SRCDIR"], "..", "00161295.48859.fit"
# )  # gps

# fits_path = os.path.join(
#     os.environ["SRCDIR"], "..", "00161341.GALAXY_23__TELSTAR_13__#27854U.fit"
# )
fits_path = os.path.join(os.environ["SRCDIR"], "..", "00161298.Jupiter.fit")


fits_dict = mr.info_from_fits(fits_path)
obs_dates = fits_dict["dates"]
observing_station = fits_dict["station"]
obs_dirs_eci = fits_dict["look_dirs_eci"]
ccd_adu = fits_dict["ccd_adu"]
br_parabola_obs = fits_dict["br_parabola"]

# %%
# Let's synthesize a CCD image for the same observation conditions

# pl = pv.Plotter()
# mrv.render_observation_scenario(
#     pl,
#     dates=obs_dates,
#     station=observing_station,
#     look_dirs_eci=obs_dirs_eci,
#     sensor_extent_km=20e3,
# )
# pl.show()

# %%
# Synthesizing the same image

# %%
# Let's synthesize a CCD image for the same observation conditions

observing_station.telescope.fwhm = 4

obj = mr.SpaceObject("matlib_hylas4.obj", identifier=26853)
r_obj_eci = obj.propagate(obs_dates)

sv = mr.sun(obs_dates)
nadir = -mr.hat(r_obj_eci)
attitude = mr.AlignedAndConstrainedAttitude(
    v_align=nadir,
    v_const=sv,
    dates=obs_dates,
    axis_order=(1, 2, 0),
)
obj_lc_sampler, _ = observing_station.observe_light_curve(
    obj,
    attitude,
    mr.Brdf("phong"),
    obs_dates,
    use_engine=True,
    instances=1,
    model_scale_factor=1,
    rotate_panels=True,
)
lc_adu = obj_lc_sampler()
print(lc_adu)
lc_adu = 1e6 * np.ones(lc_adu.shape)

catalog = mr.StarCatalog("gaia", observing_station, obs_dates[0])

mr.tic()
adu_grid_streaked_sampled = observing_station.telescope.ccd.generate_ccd_image(
    obs_dates,
    observing_station,
    obs_dirs_eci,
    lc_adu,
    catalog,
    hot_pixel_probability=0,
    dead_pixel_probability=0,
    add_parabola=False,
    scintillation=False,
)
mr.toc()


# %%
# Let's take a look at the real and synthetic CCD images

ccd_adu = np.clip(ccd_adu - br_parabola_obs, 1, np.inf)
adu_grid_streaked_sampled = np.clip(
    adu_grid_streaked_sampled - mr.image_background_parabola(adu_grid_streaked_sampled),
    1,
    np.inf,
)


plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(np.log10(ccd_adu), cmap="gray")
mrv.texit(f"POGS CCD", "", "", grid=False)
plt.colorbar(cax=mrv.get_cbar_ax(), label="$\log_{10}(ADU)$")
plt.clim(*np.percentile(np.log10(ccd_adu), [0.1, 99.9]))

plt.subplot(1, 2, 2)
plt.imshow(np.log10(adu_grid_streaked_sampled), cmap="gray")
mrv.texit(f"Synthetic CCD", "", "", grid=False)
plt.colorbar(cax=mrv.get_cbar_ax(), label="$\log_{10}(ADU)$")
plt.clim(*np.percentile(np.log10(adu_grid_streaked_sampled), [0.1, 99.9]))
plt.tight_layout()
plt.show()


# %%
# Looking at the residual noise after subtracting off the parabolic background from the real image

ccd_adu_minus_br = observing_station.telescope.ccd.subtract_parabola(ccd_adu)
real_br_mask = mr.image_background_naive(ccd_adu_minus_br)[0]
real_br_pixels = np.ma.array(ccd_adu_minus_br, mask=~real_br_mask)
synth_adu_minus_br = observing_station.telescope.ccd.subtract_parabola(
    adu_grid_streaked_sampled
)
synth_br_mask = mr.image_background_naive(synth_adu_minus_br)[0]
synth_br_pixels = np.ma.array(synth_adu_minus_br, mask=~synth_br_mask)
print(f"Real background variance: {np.var(ccd_adu_minus_br[real_br_mask])} [ADU^2]")
print(
    f"Synthetic background variance: {np.var(synth_adu_minus_br[synth_br_mask])} [ADU^2]"
)


# %%
# Plotting the same, with the parabolic background subtracted from the real image

cbar_kwargs = dict(label="$ADU$")
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(real_br_pixels, cmap="plasma")
mrv.texit(f"POGS CCD Background Pixels", "", "", grid=False)
plt.colorbar(cax=mrv.get_cbar_ax(), **cbar_kwargs)
plt.clim(0, 300)

plt.subplot(1, 2, 2)
plt.imshow(synth_br_pixels, cmap="plasma")
mrv.texit(f"Synthetic CCD Background Pixels", "", "", grid=False)
plt.colorbar(cax=mrv.get_cbar_ax(), **cbar_kwargs)
plt.clim(0, 300)
plt.tight_layout()
plt.show()

# %%
# Inspecting the backgrounds

frac_cuts = (1e-4, 5e-3)
thresh = slice(
    int(frac_cuts[0] * adu_grid_streaked_sampled.size),
    int((1 - frac_cuts[1]) * adu_grid_streaked_sampled.size),
)
synth_br_data = np.sort(adu_grid_streaked_sampled.flatten())[thresh][::100]
real_br_data = np.sort(ccd_adu.flatten())[thresh][::100]

synth_br = np.mean(synth_br_data)
real_br = np.mean(real_br_data)

print(f"Synthetic background: {synth_br} [ADU]")
print(f"Real background: {real_br} [ADU]")

synth_br_poisson_samples = np.random.poisson(synth_br, synth_br_data.size)
real_br_poisson_samples = np.random.poisson(real_br, real_br_data.size)

plt.subplot(1, 2, 2)
bins = np.arange(np.min(synth_br_data), np.max(synth_br_data))
hist_args = dict(density=True, bins=bins, alpha=0.7)
plt.hist(synth_br_data, **hist_args)
plt.hist(synth_br_poisson_samples, **hist_args)
mrv.texit("Synthetic backgrounds", "ADU", "Density", ["Image", "Poisson fit"])

plt.subplot(1, 2, 1)
hist_args["bins"] = np.arange(
    np.min(real_br_poisson_samples), np.max(real_br_poisson_samples)
)
plt.hist(real_br_data, **hist_args)
plt.hist(real_br_poisson_samples, **hist_args)
mrv.texit("Real backgrounds", "ADU", "Density", ["Image", "Poisson fit"])

plt.tight_layout()
plt.gcf().set_size_inches(8, 4)
plt.show()
