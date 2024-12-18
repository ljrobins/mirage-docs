"""
Non-Convex Inversion
====================

Implementing non-convex inversion using my method from summer 2022 :cite:p:`robinson2022`
"""

import os

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
            brdf, rec_obj_with_concavity, svb[::5, :], ovb[::5, :]
        )
        err = np.sum((lc.flatten()[::5] - lc_hat.flatten()) ** 2)
        print(f'Tried psi = {psi_deg:.1f} deg, got err = {err:.2f}')
        return err

    print('Optimizing concavity angle...')
    psis = np.arange(10, 90, 10)
    errs = np.array([objective_function(psi) for psi in psis])

    psi_opt = psis[np.argmin(errs)]
    print(f'Optimal concavity angle: {psi_opt:.1f} deg')
    return rec_convex_obj.introduce_concavity(
        err_egi_dir, psi_opt, normal_selection_tolerance=np.pi / 2 - 0.5, linear_iter=3
    )


iter_concavity = True
model_scale_factor = 0.10
itensor = np.diag([1.0, 2.0, 3.0])
w0 = 0.1 * mr.hat(np.array([[1.0, 2.0, 1.0]]))
q0 = np.array([0.0, 0.0, 0.0, 1.0])
# q0 = mr.rand_quaternions(1)
idate = mr.utc(2023, 3, 26, 5)
obs_time = mr.hours(3)
obs_dt = mr.seconds(10)
integration_time_s = obs_dt.total_seconds()
win_width = 1500
obj_kwargs = dict(opacity=0.8, feature_edges=True, feature_edge_angle=10, line_width=1)

station = mr.Station(preset='pogs')
brdf = mr.Brdf(name='phong', cd=0.5, cs=0.5, n=10)
attitude = mr.RbtfAttitude(w0=w0, q0=q0, itensor=itensor)
dates, epsecs = mr.date_arange(idate, idate + obs_time, obs_dt, return_epsecs=True)

q_of_t, w_of_t = attitude.propagate(epsecs)
dcms_of_t = mr.quat_to_dcm(q_of_t)


for noise in [False, True]:
    for non_convex in [False, True]:
        if non_convex:
            object_files = [
                'collapsed_cube.obj',
                'collapsed_ico.obj',
                'collapsed_cyl.obj',
                'collapsed_house.obj',
            ]
        else:
            object_files = ['cube.obj', 'icosahedron.obj', 'cylinder.obj', 'gem.obj']

        # mrv.vis_attitude_motion(mr.SpaceObject(object_files[0]), q_of_t)
        # endd

        station.constraints = [
            mr.SnrConstraint(3),
            mr.ElevationConstraint(15),
            mr.TargetIlluminatedConstraint(),
            mr.ObserverEclipseConstraint(station),
            mr.VisualMagnitudeConstraint(18),
            mr.MoonExclusionConstraint(30),
        ]

        pl = pv.Plotter(
            shape=(len(object_files), 4),
            window_size=(int(win_width * 4 / 3), win_width // 4 * len(object_files)),
        )
        for i, obj_file in enumerate(object_files):
            obj = mr.SpaceObject(obj_file, identifier='goes 15')
            max_vertex_disp = np.max(mr.vecnorm(obj._mesh.points))
            obj._mesh.scale(1 / max_vertex_disp, inplace=True)

            pl.subplot(i, 0)
            mrv.render_spaceobject(pl, obj, **obj_kwargs)
            pl.add_text(
                'True Object',
                font='courier',
            )

            lc_ccd_signal_sampler, aux_data = station.observe_light_curve(
                obj,
                attitude,
                brdf,
                dates,
                integration_time_s,
                use_engine=True,
                model_scale_factor=model_scale_factor,
            )

            lc_ccd_signal = lc_ccd_signal_sampler()
            lc_noisy_irrad = lc_ccd_signal / (aux_data['sint'] * integration_time_s)
            lc_noisy_unit_irrad = (
                lc_noisy_irrad
                * (aux_data['rmag_station_to_sat'] * 1e3) ** 2
                / mr.AstroConstants.sun_irradiance_vacuum
            )

            sun_body = aux_data['sun_vector_object_body']
            obs_body = aux_data['observer_vector_object_body']

            sint = aux_data['sint']
            lc_hat = aux_data['lc_clean_norm']
            constr = aux_data['all_constraints_satisfied']
            br_mean = aux_data['background_mean']
            airy_disk_pixels = aux_data['airy_disk_pixels']
            obs_to_moon = aux_data['obs_to_moon']
            lc_clean = aux_data['lc_clean']
            snr = aux_data['snr']

            svb_masked = sun_body[~lc_noisy_unit_irrad.mask]
            ovb_masked = obs_body[~lc_noisy_unit_irrad.mask]
            lc_noisy_unit_irrad_masked = lc_noisy_unit_irrad[~lc_noisy_unit_irrad.mask]

            plt.scatter(epsecs, lc_ccd_signal)
            mrv.texit(
                f"{'Noisy' if noise else ''} Light Curve: {obj_file[:-4]} {'SNR = ' + str(np.round(np.mean(snr),2)) if noise else ''}",
                'Epoch seconds',
                'ADU',
                grid=False,
            )
            plt.ylim(0, 1.1 * np.max(lc_ccd_signal))
            plt.tight_layout()
            plt.gcf().savefig('temp.png', format='png', dpi=180)
            plt.clf()

            pl.subplot(i, 1)
            pl.add_background_image('temp.png', as_global=False)
            os.remove('temp.png')

            # %%
            # Inversion

            # Inversion
            egi_opt_initial, egi_opt = mr.optimize_egi(
                (
                    lc_noisy_unit_irrad_masked
                    if noise
                    else lc_hat[~lc_noisy_unit_irrad.mask]
                ),
                svb_masked,
                ovb_masked,
                brdf,
                merge_iter=2,
                merge_angle=np.pi / 8,
                return_all=True,
                num_candidates=2000,
            )

            # Inversion
            rec_obj = mr.construct_mesh_from_egi(mr.close_egi(egi_opt))

            # %%
            # Introducing the concavity

            err_egi = -np.sum(egi_opt_initial, axis=0)
            err_egi_mag = np.linalg.norm(err_egi)
            err_egi_dir = mr.hat(err_egi)

            if iter_concavity:
                rec_obj_with_concavity = match_concavity_to_light_curve(
                    rec_obj,
                    err_egi_dir,
                    brdf,
                    svb_masked,
                    ovb_masked,
                    lc_noisy_unit_irrad_masked,
                )
            else:
                rec_obj_with_concavity = rec_obj.introduce_concavity(
                    err_egi_dir,
                    45,
                    linear_iter=3,
                    normal_selection_tolerance=np.pi / 2 - 0.5,
                )

            # %%
            # Measuring error in reconstructed convex and nonconvex objects

            lc_hat = mr.run_light_curve_engine(
                brdf, rec_obj, svb_masked[::10, :], ovb_masked[::10, :]
            )
            err_cvx = np.sum(
                (lc_noisy_unit_irrad_masked.flatten()[::10] - lc_hat.flatten()) ** 2
            )

            lc_hat = mr.run_light_curve_engine(
                brdf, rec_obj_with_concavity, svb_masked[::10, :], ovb_masked[::10, :]
            )
            err_ncvx = np.sum(
                (lc_noisy_unit_irrad_masked.flatten()[::10] - lc_hat.flatten()) ** 2
            )

            ncvx_better = err_cvx > err_ncvx

            pl.subplot(i, 2)
            mrv.render_spaceobject(
                pl,
                rec_obj,
                **obj_kwargs,
                color='lightcoral' if ncvx_better else 'palegreen',
            )
            pl.add_text(
                'Convex Guess',
                font='courier',
            )
            pl.subplot(i, 3)
            mrv.render_spaceobject(
                pl,
                rec_obj_with_concavity,
                **obj_kwargs,
                color='palegreen' if ncvx_better else 'lightcoral',
            )

            pl.add_text(
                'Non-Convex Guess',
                font='courier',
            )

        pl.show()
