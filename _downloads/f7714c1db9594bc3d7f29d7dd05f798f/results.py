"""
Results
=======
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyvista as pv
import vtk  # noqa

import mirage as mr
import mirage.vis as mrv

f = '/Users/liamrobinson/Documents/maintained-research/mirage/examples/15-attitude-sdc-2025/data/'

dfs_delta = (
    pl.read_parquet(os.path.join(f, 'delta_in.parquet')),
    pl.read_parquet(os.path.join(f, 'delta_out_better_mat4.parquet')).sort('fun'),
)

dfs_echo = (
    pl.read_parquet(os.path.join(f, 'echo_in.parquet')),
    pl.read_parquet(os.path.join(f, 'echo_out_better_mat4.parquet')).sort('fun'),
)

dfs_synth = [(dfs_delta, 'delta', 1 / 100), (dfs_echo, 'echo', 1 / 1000)]
cdist = 8.0
bad_sol_opacity = 0.4
window_size = (900, 1000)

for (df_in, df), label, bound in dfs_synth:
    for c in ['x0', 'xk']:
        x = df[c].to_numpy().copy()
        x[:, 3:6] /= df_in['epsecs'].max()
        df = df.with_columns(pl.Series(name=c, values=x))

    print(
        f'{label} max search omega in deg/s',
        np.rad2deg(
            np.linalg.norm(df['x0'].to_numpy()[:, 3:6], axis=1, keepdims=True).mean()
        ),
    )

    df_sols = df.with_columns((-pl.col('fun')).exp().alias('likelihood'))
    df_sols = df_sols.filter(pl.col('likelihood') > bound * df_sols['likelihood'][0])

    print(f'{df_sols.height} solutions found meeting {bound=}')

    x0 = np.array(df['x0'].to_list())
    xk = np.array(df['xk'].to_list())

    xk_sol = np.array(df_sols['xk'].to_list())
    xk[:, 6:] = np.abs(xk[:, 6:])
    xk_sol[:, 6:] = np.abs(xk_sol[:, 6:])

    ang_traversed = np.abs(
        np.rad2deg(mr.quat_ang(mr.mrp_to_quat(x0[:, :3]), mr.mrp_to_quat(xk[:, :3])))
    )

    plt.figure()
    rnge = np.percentile(xk_sol[:, 6:], axis=0, q=[5, 95]).T
    h = plt.hist2d(*xk_sol[:, 6:].T, range=rnge, bins=40, cmap='viridis')[-1]
    # plt.scatter(*xk_sol[:, 6:].T, c='r', marker='*', label='Candidate solutions')
    plt.scatter(*x0[0, 6:], c='r', marker='x', label='Initial estimate')
    plt.colorbar(h, label='Frequency of converged local minima', cax=mrv.get_cbar_ax())
    plt.xlabel('Inertia ratio $J_y J_x^{-1}$')
    plt.ylabel('Inertia ratio $J_z J_x^{-1}$')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    p = pv.Plotter(window_size=window_size)
    # mrv.scatter3(p, np.rad2deg(xk[:, 3:6]), opacity=bad_sol_opacity)
    mrv.scatter3(p, np.rad2deg(xk_sol[:, 3:6]), point_size=25, color='gray')
    lim = np.percentile(np.linalg.norm(np.rad2deg(x0[:, 3:6]), axis=1), 95)

    p.show_grid(
        # xtitle='$\mathbf{\omega_{x,0}}$ [deg/s]',
        # ytitle='$\mathbf{\omega_{y,0}}$ [deg/s]',
        # ztitle='$\mathbf{\omega_{z,0}}$ [deg/s]',
        font_family='times',
        font_size=10,
        bounds=np.array([-1, 1, -1, 1, -1, 1]) * lim,
    )
    p.camera.position = (
        cdist * lim * np.array(p.camera.position) / np.linalg.norm(p.camera.position)
    )
    mrv.orbit_plotter(p, video_name=f'{label}_ang_vel.mp4')
    p.show()
    p.screenshot(
        f'/Users/liamrobinson/Documents/maintained-research/mirage/examples/15-attitude-sdc-2025/figs/{label}_ang_vel.png'
    )
    p = pv.Plotter(window_size=window_size)
    mrv.scatter3(
        p,
        mr.quat_to_mrp(mr.quat_upper_hemisphere(mr.mrp_to_quat(xk_sol[:, :3]))),
        point_size=25,
        color='c',
    )
    p.show_grid(
        # xtitle='$\\mathbf{p}_{1}$(0) [nondim]',
        # ytitle='$\\mathbf{p}_{2}$(0) [nondim]',
        # ztitle='$\\mathbf{p}_{3}$(0) [nondim]',
        font_family='times',
        font_size=12,
        bounds=np.array([-1, 1, -1, 1, -1, 1]),
    )
    p.camera.position = (
        cdist * np.array(p.camera.position) / np.linalg.norm(p.camera.position)
    )
    mrv.orbit_plotter(p, video_name=f'{label}_sigma.mp4')
    p.show()
    # endd

    lc_obs = df_in['flux_adu'].to_numpy().flatten()
    sigma_obs_counts = df_in['sigma_obs_counts'].to_numpy().flatten()

    plt.figure(figsize=(10, 3))

    for i in range(min(10, df_sols.height)):
        d = df_sols[i].to_dict()

        t = df_in['epsecs'].to_numpy().flatten() / 60
        lc_est = d['lcs'].to_numpy().flatten()

        print([np.round(x, 2) for x in d['xk'].to_list()[0]])
        print(d['x0'][0][6:])

        lc_est = (
            lc_est / np.linalg.norm(lc_est) * np.linalg.norm(lc_obs[~np.isnan(lc_obs)])
        )

        # for i in d.keys():
        #     print(f'{i}: {d[i]}')

        plt.plot(
            t,
            lc_est,
            'm-',
            alpha=0.5,
            linewidth=1.0,
            label='Best solutions' if i == 0 else None,
        )
    plt.plot(t, lc_obs, 'k+-', label='Observations')

    plt.fill_between(
        t,
        np.clip(lc_obs - 3 * sigma_obs_counts, 0, np.inf),
        np.clip(lc_obs + 3 * sigma_obs_counts, 0, np.inf),
        alpha=0.4,
        color='gray',
        edgecolor=None,
        label='Computed $\pm3\sigma$',
    )

    plt.xlabel('Minutes past observation epoch')
    plt.ylabel('Observed flux [ADU]')
    plt.grid()
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.show()
