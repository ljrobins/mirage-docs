"""
Non-Convex Inversion
====================

Implementing non-convex inversion using my method from summer 2022 :cite:p:`robinson2022`
"""

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv


def match_concavity_to_light_curve(
    rec_convex_obj: mr.SpaceObject,
    err_egi_dir: np.ndarray,
    brdf: mr.Brdf,
    svb: np.ndarray,
    ovb: np.ndarray,
    lc: np.ndarray,
) -> mr.SpaceObject:
    def objective_function(psi_deg: float) -> float:
        rec_obj_with_concavity = rec_convex_obj.introduce_concavity(
            err_egi_dir,
            psi_deg,
            linear_iter=3,
            normal_selection_tolerance=np.pi / 2 - 0.5,
        )
        lc_hat = mr.run_light_curve_engine(
            brdf, rec_obj_with_concavity, svb[::10, :], ovb[::10, :]
        )
        lc_hat /= np.max(lc_hat)
        err = np.sum((lc.flatten()[::10] - lc_hat.flatten()) ** 2)
        print(f"Tried psi = {psi_deg:.1f} deg, got err = {err:.2f}")
        return err

    lc /= np.max(lc)
    print("Optimizing concavity angle...")
    psis = np.arange(0, 90, 5)
    errs = np.array([objective_function(psi) for psi in psis])

    psi_opt = psis[np.argmin(errs)]
    print(f"Optimal concavity angle: {psi_opt:.1f} deg")
    return rec_convex_obj.introduce_concavity(
        err_egi_dir, psi_opt, normal_selection_tolerance=np.pi / 2 - 0.5
    )


w0_mag = 1e-1
itensor = np.diag([1.0, 2.0, 3.0])
w0 = w0_mag * mr.hat(np.array([[1.0, 2.0, 1.0]]))
idate = mr.utc(2023, 1, 1, 5)
obs_time = mr.days(1)
obs_dt = mr.seconds(10)

object_files = [
    "collapsed_cube.obj",
    "collapsed_ico.obj",
    "collapsed_cyl.obj",
    "collapsed_house.obj",
]

station = mr.Station(preset="pogs")
brdf = mr.Brdf(name="phong", cd=0.5, cs=0.5, n=10)
attitude = mr.RbtfAttitude(w0=w0, q0=mr.rand_quaternions(1), itensor=itensor)
dates, epsecs = mr.date_arange(idate, idate + obs_time, obs_dt, return_epsecs=True)

q_of_t, w_of_t = attitude.propagate(epsecs)
dcms_of_t = mr.quat_to_dcm(q_of_t)

station.constraints = [
    mr.SnrConstraint(3),
    mr.ElevationConstraint(10),
    mr.TargetIlluminatedConstraint(),
    mr.ObserverEclipseConstraint(station),
    mr.VisualMagnitudeConstraint(18),
    mr.MoonExclusionConstraint(10),
    mr.HorizonMaskConstraint(station),
]

win_width = 1500
obj_kwargs = dict(opacity=0.8, feature_edges=True)
pl = pv.Plotter(
    shape=(len(object_files), 4), window_size=(int(win_width * 4 / 3), win_width)
)
for i, obj_file in enumerate(object_files[:4]):
    obj = mr.SpaceObject(obj_file, identifier="goes 15")
    max_vertex_disp = np.max(mr.vecnorm(obj._mesh.points))
    obj._mesh.scale(1 / max_vertex_disp, inplace=True)

    pl.subplot(i, 0)
    mrv.render_spaceobject(pl, obj, **obj_kwargs)
    pl.add_text(
        f"True Object",
        font="courier",
    )

    lc_ccd_signal_sampler, aux_data = station.observe_light_curve(
        obj, attitude, brdf, dates, use_engine=True, model_scale_factor=100
    )

    lc_ccd_signal = lc_ccd_signal_sampler()
    lc_noisy_irrad = lc_ccd_signal / (
        aux_data["sint"] * station.telescope.integration_time
    )
    lc_noisy_unit_irrad = lc_noisy_irrad * (aux_data["rmag_station_to_sat"] * 1e3) ** 2
    lc_noisy_unit_irrad /= np.max(lc_noisy_unit_irrad)

    plt.scatter(epsecs, lc_ccd_signal)
    mrv.texit(f"Light Curve: {obj_file}", "Epoch seconds", "ADU", grid=False)
    plt.gcf().savefig("temp.png", format="png", dpi=180)
    plt.clf()

    pl.subplot(i, 1)
    pl.add_background_image("temp.png", as_global=False)

    # %%
    # Inversion

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

    # Inversion
    # brdf_for_inversion = mr.Brdf("phong", cd=0.5, cs=0.0, n=10)
    brdf_for_inversion = brdf
    egi_opt_initial, egi_opt = mr.optimize_egi(
        lc_noisy_unit_irrad[~lc_noisy_unit_irrad.mask],
        sun_body[~lc_noisy_unit_irrad.mask, :],
        obs_body[~lc_noisy_unit_irrad.mask, :],
        brdf_for_inversion,
        merge_iter=1,
        merge_angle=np.pi / 6,
        return_all=True,
    )

    # Inversion
    h_opt = mr.optimize_supports_little(egi_opt)
    rec_obj = mr.construct_from_egi_and_supports(egi_opt, h_opt)

    # Plotting inverted result
    pl.subplot(i, 2)
    mrv.render_spaceobject(pl, rec_obj, **obj_kwargs)
    pl.add_text(
        f"Convex Guess",
        font="courier",
    )

    # %%
    # Introducing the concavity

    err_egi = -np.sum(egi_opt_initial, axis=0)
    err_egi_mag = np.linalg.norm(err_egi)
    err_egi_dir = mr.hat(err_egi)

    rec_obj_with_concavity = match_concavity_to_light_curve(
        rec_obj,
        err_egi_dir,
        brdf,
        sun_body[~lc_noisy_unit_irrad.mask],
        obs_body[~lc_noisy_unit_irrad.mask],
        lc_noisy_unit_irrad[~lc_noisy_unit_irrad.mask],
    )

    # rec_obj_with_concavity = rec_obj.introduce_concavity(err_egi_dir, 45, linear_iter=3, normal_selection_tolerance=np.pi/2 - 0.5)

    pl.subplot(i, 3)
    mrv.render_spaceobject(pl, rec_obj_with_concavity, **obj_kwargs)

    pl.add_text(
        f"Non-Convex Guess",
        font="courier",
    )


pl.show()
