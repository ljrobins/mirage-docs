"""
Inversion PDF
=============

Estimating the probability density function for the surface of the object
"""

import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

nper = 10
height = 500
model_scale_factor = [0.2, 0.1, 0.05, 0.04]
pl = pv.Plotter(
    shape=(1, len(model_scale_factor)),
    window_size=(height * len(model_scale_factor), height),
)

itensor = np.diag([1.0, 2.0, 3.0])
w0 = 0.1 * mr.hat(np.array([[1.0, 2.0, 1.0]]))
q0 = np.array([0.0, 0.0, 0.0, 1.0])
idate = mr.utc(2023, 3, 26, 10)
obs_time = mr.hours(3)
obs_dt = mr.seconds(10)
integration_time_s = obs_dt.total_seconds()

obj = mr.SpaceObject('cube.obj', identifier='goes 15')
station = mr.Station(preset='pogs')
brdf = mr.Brdf(name='phong', cd=0.5, cs=0.5, n=10)
attitude = mr.RbtfAttitude(w0=w0, q0=q0, itensor=itensor)
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
]

for i, msf in enumerate(model_scale_factor):
    pl.subplot(0, i)
    lc_ccd_signal_sampler, aux_data = station.observe_light_curve(
        obj,
        attitude,
        brdf,
        dates,
        integration_time_s,
        use_engine=True,
        model_scale_factor=msf,
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
    mean_snr = np.mean(snr)

    pl.add_text(
        f'Width {2*msf:.2f} m\nSNR = {mean_snr:.1f}',
        font_size=12,
        font='courier',
        position='upper_left',
        color='k',
    )

    rec_objs = []
    for _ in range(nper):
        lc_ccd_signal = lc_ccd_signal_sampler()
        lc_noisy_irrad = lc_ccd_signal / (aux_data['sint'] * integration_time_s)
        lc_noisy_unit_irrad = (
            lc_noisy_irrad
            * (aux_data['rmag_station_to_sat'] * 1e3) ** 2
            / mr.AstroConstants.sun_irradiance_vacuum
        )

        egi_opt_initial, egi_opt = mr.optimize_egi(
            lc_noisy_unit_irrad[~lc_noisy_unit_irrad.mask],
            sun_body[~lc_noisy_unit_irrad.mask, :],
            obs_body[~lc_noisy_unit_irrad.mask, :],
            brdf,
            merge_iter=1,
            merge_angle=np.pi / 6,
            return_all=True,
            num_candidates=1000,
        )
        rec_obj = mr.construct_mesh_from_egi(mr.close_egi(egi_opt))
        rec_objs.append(rec_obj)

    grid = mr.r3_grid(1.2 * np.max(mr.vecnorm(obj.v)), 150)

    for i, rec_obj in enumerate(rec_objs):
        rec_obj.file_name = f'rec_obj_{i}.obj'
        mrv.render_spaceobject(
            pl, rec_obj, opacity=0.3, feature_edges=True, line_width=2
        )
        mrv.scatter3(pl, rec_obj.v, color='k', point_size=5)
    pl.disable_anti_aliasing()
    pl.view_isometric()

# pl.link_views()
pl.show()
