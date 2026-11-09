"""
Results
=======
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

# from matplotlib import colormaps
# from matplotlib.colors import SymLogNorm
import mirage as mr
import mirage.vis as mrv

# from scipy.interpolate import RegularGridInterpolator

np.random.seed(2)

station = mr.Station()

obj_path = os.path.join(os.environ['SRCDIR'], 'resources/models/matlib_box_wing.obj')

obj = mr.load_obj(obj_path, identifier=25544)

kwargs = dict(
    noise_counts_per_sec=int(1e3),
    trajectory_time_resolution=mr.seconds(1),
    range_gate_ns=100,
    depth_bin_count=100,
    njsd_tol=0.01,
    shgo_iters=5,
)

alpha = 0.2
attitude = mr.RbtfAttitude(
    w0=np.deg2rad(np.array([0, 0, 7])),
    q0=np.array([0, 0, 0, 1.0]),
    itensor=np.diag([1, 2, 3]),
)
brdf = mr.Brdf('blinn-phong', cd=obj.cd, cs=obj.cs, n=1000 * obj.n, validate=False)

laser = mr.Laser(
    hz=100,
    lam=1064,
    pulse_energy=200e-3,
    beam_divergence_half_angle=mr.dms_to_rad(0, 0, 15),
    tracking_jitter_rad=mr.dms_to_rad(0, 0, 1),
    beam_pointing_error_rad=mr.dms_to_rad(0, 0, 3),
    pulse_fwhm_ns=1,
)
start_date = mr.utc(2025, 6, 27, 7, 47)
end_date = mr.utc(2025, 6, 27, 7, 57)

P_H = 1000
T_H = 300
R_H = 50

## DIFFERENT SHAPE, DIFFERENT ATTITUDE

plt.figure(figsize=(13, 4))


objs = [
    (
        mr.load_obj(
            os.path.join(
                os.environ['SRCDIR'],
                '..',
                'examples/15-attitude-sdc-2025/models/star37e.obj',
            ),
            identifier=25544,
        ),
        'Delta I R/B',
    ),
    (obj, 'Generic Box-Wing'),
    (
        mr.load_obj(
            os.path.join(os.environ['SRCDIR'], 'resources/models/matlib_gps_iii.obj'),
            identifier=25544,
        ),
        'GPS-III',
    ),
]


p = pv.Plotter(shape=(1, 3), window_size=(1500, 500))

for i, (_obj, name) in enumerate(objs):
    _obj.shift_to_center_of_mass()
    p.subplot(0, i)
    mrv.render_spaceobject(p, _obj)
    mrv.plot_basis(
        p,
        np.eye(3),
        labels=['x', 'y', 'z'],
        scale=np.max(mr.vecnorm(_obj.v)) / 2,
        shape_opacity=0.5,
    )
    p.add_text(name)
p.show()

for i, (_obj, name) in enumerate(objs, 1):
    _brdf = mr.Brdf(
        'blinn-phong', cd=_obj.cd, cs=_obj.cs, n=1000 * _obj.n, validate=False
    )

    # attitude.q0 = mr.rand_quaternions(1) # time to regen
    # attitude.w0 = mr.rand_unit_vectors(1) * np.deg2rad(10)

    df = mr.sdlr_residuals_adaptive(
        station, _obj, attitude, _brdf, laser, start_date, end_date, **kwargs
    )

    plt.subplot(1, 3, i)
    colors_map = np.array(['black', 'red'])

    # Apply the mapping
    cols = colors_map[df['is_noise'].to_numpy().astype(int)]

    plt.scatter(df['epsecs'], df['est_depth_poly'], s=1, alpha=alpha, c=cols)
    plt.ylim(df['est_depth_poly'].quantile(0.01), df['est_depth_poly'].quantile(0.99))
    plt.title(name)
    plt.xlabel('Seconds')
    plt.ylabel('O-C [m]')
    plt.grid()

plt.tight_layout()
plt.show()

## SAME SHAPE, DIFFERENT ATTITUDE

plt.figure(figsize=(13, 4))

for i in range(1, 4):
    laser.hz = 100
    attitude.q0 = mr.rand_quaternions(1)
    attitude.w0 = mr.rand_unit_vectors(1) * np.deg2rad(10)

    df = mr.sdlr_residuals_adaptive(
        station, obj, attitude, brdf, laser, start_date, end_date, **kwargs
    )

    plt.subplot(1, 3, i)
    cols = plt.get_cmap('gnuplot')(df['is_noise'])
    plt.scatter(df['epsecs'], df['est_depth_poly'], s=1, alpha=alpha, c=cols)
    # plt.scatter(df['epsecs'], df['est_depth'], s=1, alpha=alpha)
    plt.ylim(df['est_depth_poly'].quantile(0.01), df['est_depth_poly'].quantile(0.99))
    plt.title(f'Attitude profile {i}')
    plt.xlabel('Seconds')
    plt.ylabel('O-C [m]')
    plt.grid()

plt.tight_layout()
plt.show()

## LINEAR, PULSE RATE VS ENERGY, CONSTANT POWER

plt.figure(figsize=(13, 3))

pulse_hz = [10, 100, 400, 1000]
watt_output = 20

i = 0
for hz in pulse_hz:
    i += 1
    laser.hz = hz
    laser.pulse_energy = watt_output / hz
    print(laser.hz, laser.pulse_energy)

    df = mr.sdlr_residuals_adaptive(
        station, obj, attitude, brdf, laser, start_date, end_date, **kwargs
    )

    plt.subplot(1, len(pulse_hz), i)
    cols = plt.get_cmap('gnuplot')(df['is_noise'])
    plt.scatter(df['epsecs'], df['est_depth_poly'], s=1, alpha=alpha, c=cols)
    plt.ylim(df['est_depth_poly'].quantile(0.01), df['est_depth_poly'].quantile(0.99))
    plt.title(f'{hz} [Hz], {int(laser.pulse_energy * 1e3)} [mJ]')
    plt.xlabel('Seconds')
    plt.ylabel('O-C [m]')
    plt.grid()

plt.tight_layout()
plt.show()

## GRID OF PULSE LENGTH VS TARGET SIZE

laser.hz = 100
laser.pulse_energy = 200e-3
plt.figure(figsize=(11, 8))

target_sizes = [0.5, 1.0, 3.0]  # m, largest axis
pulse_widths = [1.0, 3.0, 7.0]  # ns

i = 0
for ts in target_sizes:
    for pw in pulse_widths:
        i += 1
        laser.pulse_fwhm_ns = pw
        obj.v /= mr.vecnorm(obj.v).max()
        obj.v *= ts
        obj.build_properties()

        df = mr.sdlr_residuals_adaptive(
            station, obj, attitude, brdf, laser, start_date, end_date, **kwargs
        )

        plt.subplot(3, 3, i)
        cols = plt.get_cmap('gnuplot')(df['is_noise'])
        plt.scatter(df['epsecs'], df['est_depth_poly'], s=1, alpha=alpha, c=cols)
        plt.ylim(
            df['est_depth_poly'].quantile(0.01), df['est_depth_poly'].quantile(0.99)
        )
        plt.title(f'Size: {ts} [m], FWHM {pw} [ns]')
        plt.xlabel('Seconds')
        plt.ylabel('O-C [m]')
        plt.grid()

plt.tight_layout()
plt.show()
