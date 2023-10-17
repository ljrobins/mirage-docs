"""
Light Curve Inversion with Uncertainty
======================================
Inverting convex shapes from light curves with uncertainty
"""

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

itensor = np.diag([1.0, 2.0, 3.0])
w0 = 9e-2 * mr.hat(np.array([[1.0, 1.0, 1.0]]))
idate = mr.utc(2023, 1, 1, 5)
obs_time = mr.hours(3)
obs_dt = mr.seconds(10)
pl_shape = (4, 4)
nights = np.prod(pl_shape)

obj_file = "cylinder.obj"

station = mr.Station(preset="pogs")
brdf = mr.Brdf(name="phong", cd=0.5, cs=0.5, n=10)
brdf_inversion = brdf
# brdf_inversion = mr.Brdf(name='phong', cd=1, cs=0.0, n=5)
attitude = mr.RbtfAttitude(w0=w0, q0=np.array([[0.0, 0.0, 0.0, 1.0]]), itensor=itensor)

dates = []
epsecs = []
for i in range(nights):
    d = mr.date_arange(idate, idate + obs_time, obs_dt)
    dates.append(d)
    idate += mr.days(1)
dates = np.concatenate(dates)
epsecs = np.array([(d - dates[0]).total_seconds() for d in dates])

q_of_t, w_of_t = attitude.propagate(epsecs)
dcms_of_t = mr.quat_to_dcm(q_of_t)
model_scale_factor = 0.1

obj = mr.SpaceObject(obj_file, identifier="goes 15")
obj.shift_to_center_of_mass()

lc_ccd_signal_sampler, aux_data = station.observe_light_curve(
    obj, attitude, brdf, dates, use_engine=False, model_scale_factor=model_scale_factor
)

sun_body = aux_data["sun_vector_object_body"]
obs_body = aux_data["observer_vector_object_body"]

sint = aux_data["sint"]
lc_hat = aux_data["lc_clean_norm"]
constr = aux_data["all_constraints_satisfied"]
br_mean = aux_data["background_mean"]
airy_disk_pixels = aux_data["airy_disk_pixels"]
obs_to_moon = aux_data["obs_to_moon"]
lc_clean = aux_data["lc_clean"]
snr = aux_data["snr"]

# plt.figure(figsize=(7, 5))
lcs_noisy_adu = np.array([lc_ccd_signal_sampler() for _ in range(1000)])
lcs_noisy_irrad = lcs_noisy_adu / (
    aux_data["sint"] * station.telescope.integration_time
)
lcs_noisy_unit_irrad = (
    lcs_noisy_irrad
    * (aux_data["rmag_station_to_sat"] * 1e3) ** 2
    / mr.AstroConstants.sun_irradiance_vacuum
)
# lcs_noisy_mag = mr.irradiance_to_apparent_magnitude(lcs_noisy_irrad)
# var_lcs = np.var(lcs_noisy_mag, axis=0)
# mean_lcs = np.mean(lcs_noisy_mag, axis=0)

# plt.plot(epsecs, mean_lcs, c="k")
# for stdev in [1, 2, 3]:
#     plt.fill_between(
#         epsecs,
#         mean_lcs - (stdev - 1) * np.sqrt(var_lcs),
#         mean_lcs - stdev * np.sqrt(var_lcs),
#         alpha=0.4 - 0.1 * stdev,
#         color="b",
#         edgecolor=None,
#     )
#     plt.fill_between(
#         epsecs,
#         mean_lcs + (stdev - 1) * np.sqrt(var_lcs),
#         mean_lcs + stdev * np.sqrt(var_lcs),
#         alpha=0.4 - 0.1 * stdev,
#         color="b",
#         edgecolor=None,
#     )
# mrv.texit(
#     "Light Curve with Uncertainty",
#     "Epoch seconds",
#     "Apparent Magnitude",
#     grid=False,
#     legend=["Mean", "1$\sigma$", "2$\sigma$", "3$\sigma$"],
# )
# plt.show()

pl = pv.Plotter(shape=pl_shape)
rec_objs = []
lc = lcs_noisy_unit_irrad[0, :]
for i in range(nights):
    inds = (np.array([i, i + 1]) / nights * lc.size).astype(int)

    lc_this = lc[inds[0] : inds[1]]
    sun_body_this = sun_body[inds[0] : inds[1], :]
    obs_body_this = obs_body[inds[0] : inds[1], :]

    egi_opt_initial, egi_opt = mr.optimize_egi(
        lc_this / np.max(lc_this),
        sun_body_this,
        obs_body_this,
        brdf_inversion,
        merge_iter=2,
        merge_angle=np.pi / 8,
        return_all=True,
    )

    # Inversion
    h_opt = mr.optimize_supports_little(egi_opt)
    rec_obj = mr.construct_from_egi_and_supports(
        egi_opt, h_opt * np.sqrt(np.max(lc_this)) / model_scale_factor
    )
    rec_obj.shift_to_center_of_mass()
    rec_objs.append(rec_obj)


for i, rec_obj in enumerate(rec_objs):
    pl.subplot(i // pl_shape[0], i % pl_shape[1])
    mrv.render_spaceobject(pl, rec_obj)
    mrv.render_spaceobject(pl, obj, style="wireframe", color="r")
    pl.add_text(
        f"Night {i+1}",
        font="courier",
    )
pl.show()

# %%
# Measuring the error of each object's light curve across all data
errs = []
for rec_obj in rec_objs:
    lc_hat_engine = (
        mr.run_light_curve_engine(brdf, rec_obj, sun_body, obs_body)
        * model_scale_factor**2
    )
    # lc_hat = rec_obj.convex_light_curve(brdf, sun_body, obs_body) * model_scale_factor ** 2
    errs.append(np.linalg.norm(lc_hat_engine - lc))
    print(f"Error: {errs[-1]:.2e}")
errs = np.array(errs)
weights = 1 - (errs - np.min(errs)) / (np.max(errs) - np.min(errs))

# %%
# Merging the guesses and testing new error

merged_obj = mr.merge_shapes(rec_objs, weights)

pl = pv.Plotter()
mrv.render_spaceobject(pl, merged_obj, opacity=0.7)
mrv.render_spaceobject(pl, obj, style="wireframe", color="r")
# mrv.render_spaceobject(pl, rec_objs[0], style="wireframe", color="b")
pl.show()

lc_hat = (
    mr.run_light_curve_engine(brdf, merged_obj, sun_body, obs_body)
    * model_scale_factor**2
)
err = np.linalg.norm(lc_hat - lc)
print(f"New Error: {err:.2e}")
