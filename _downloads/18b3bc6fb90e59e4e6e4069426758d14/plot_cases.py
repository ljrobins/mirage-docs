"""
Plotting Cases
==============
"""

# isort: skip_file
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import mirage as mr


df_in = pl.read_parquet('examples/13-attitude-2024/data/rocket_body_input.parquet')
df_out = pl.read_parquet(
    'examples/13-attitude-2024/data/rocket-body-known-itensor-out.parquet'
)

# df_in = pl.read_parquet('examples/13-attitude-2024/data/box_wing_input.parquet')
# df_out = pl.read_parquet(
#     'examples/13-attitude-2024/data/box-wing-known-itensor-out.parquet'
# )

lc_mean = df_in['lc_true_mean_norm'].to_numpy()
lc_noisy = df_in['lc_true_noisy_norm'].to_numpy()
sigma_obs = df_in['sigma_obs_norm'].to_numpy()
epsecs = df_in['epsecs'].to_numpy()
state0true = df_in['state0'][0].to_numpy()
q0_true = mr.quat_upper_hemisphere(mr.mrp_to_quat(state0true[:3]))
s0 = mr.quat_to_mrp(q0_true)
w0 = state0true[3:6] / (epsecs.max() + np.diff(epsecs[:2]))
iterms = state0true[6:]
print(df_out)

df_conv = df_out.filter(pl.col('fun') < 1.2 * df_out['fun'][0])
n_plot = df_conv.height

if len(df_conv['xk'][0].to_list()) == 8:
    itensor_sols = np.array(df_conv['xk'].to_list())[:, -2:]
    hist = plt.hist2d(*itensor_sols.T, bins=20)[-1]
    plt.scatter(*state0true[-2:], facecolors='r', s=5, label='Truth')
    plt.xlabel('Inertia ratio $J_y \: J_x^{-1}$')
    plt.ylabel('Inertia ratio $J_z \: J_x^{-1}$')
    plt.colorbar(hist, label='Solution count')
    plt.legend()
    plt.tight_layout()
    plt.show()


xk = np.array(df_conv['xk'].to_list())
wk = xk[:, 3:6] / (epsecs.max() + np.diff(epsecs[:2]))
q0_est = mr.quat_upper_hemisphere(mr.mrp_to_quat(xk[:, :3]))
ang_err_deg = np.rad2deg(mr.quat_ang(q0_true, q0_est))
sk = mr.quat_to_mrp(q0_est)
ang_vel_az, ang_vel_el, ang_vel_mag = mr.cart_to_sph(*wk.T)
ang_vel_az_true, ang_vel_el_true, ang_vel_mag_true = mr.cart_to_sph(*w0.T)
ang_vel_mag_error_percent = (ang_vel_mag - ang_vel_mag_true) / ang_vel_mag_true * 100

plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 1)
plt.fill_between(
    epsecs,
    lc_mean - 2 * sigma_obs,
    lc_mean + 2 * sigma_obs,
    alpha=0.2,
    label='True uncertainty $\pm 2 \sigma$',
)
plt.plot(np.nan, np.nan, alpha=1, label='Best solutions', color='m')
for i, sol in enumerate(df_out.iter_rows(named=True)):
    if i == n_plot:
        break
    x0 = np.array(sol['xk'])
    lc = np.array(sol['lcs'])

    plt.plot(epsecs, lc, alpha=0.1, color='m')
plt.plot(epsecs, lc_mean, label='True mean', color='k')
plt.scatter(epsecs, lc_noisy, label='Observations', marker='+', color='k')
plt.legend()
plt.grid()
plt.xlabel('Seconds after first observations')
plt.ylabel('Normalized Brightess')
plt.ylim(0, 1.3 * plt.ylim()[1])
plt.subplot(1, 3, 2)
hist = plt.hist2d(
    ang_err_deg,
    ang_vel_mag_error_percent.flatten(),
    bins=20,
)
plt.xlabel('Orientation angle error [deg]')
plt.ylabel('Angular velocity error [%]')
plt.colorbar(label='Converged solution count')

plt.subplot(1, 3, 3)
hist = plt.hist2d(
    np.rad2deg(ang_vel_az),
    np.rad2deg(ang_vel_el),
    bins=20,
)[-1]
plt.scatter(
    np.rad2deg(ang_vel_az_true),
    np.rad2deg(ang_vel_el_true),
    facecolors='r',
    label='Truth',
)
plt.xlabel('$\omega(0)$ azimuth [deg]')
plt.ylabel('$\omega(0)$ elevation [deg]')
plt.colorbar(hist, label='Converged solution count')
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Plotting the whole 6d solution

# p = pv.Plotter()
# mrv.scatter3(p, sk, point_size=20, scalars=mr.hat(wk), rgb=True)
# p.show()
# edndd

# %%
# Plotting the angular velocity
# p = pv.Plotter(window_size=(1200, 1000))
# mrv.scatter3(
#     p,
#     wk,
#     point_size=10,
#     scalars=ang_vel_mag_error_percent,
#     clim=[ang_vel_mag_error_percent.min(), ang_vel_mag_error_percent.max()],
#     scalar_bar_args=dict(
#         title='$\|\mathbf{\omega}\|_2$ error (%)',
#         vertical=True,
#         position_x=0.85,
#         position_y=0.2,
#         height=0.5,
#         font_family='times',
#         title_font_size=30,
#         label_font_size=30,
#     ),
# )
# mrv.scatter3(p, w0, point_size=30, color='r', label='')
# p.show_grid(
#     xtitle='Body frame $\mathbf{\omega_{x,0}}$ [rad/s]',
#     ytitle='Body frame $\mathbf{\omega_{y,0}}$ [rad/s]',
#     ztitle='Body frame $\mathbf{\omega_{z,0}}$ [rad/s]',
#     font_family='times',
#     font_size=15,
# )
# mrv.two_sphere(p, radius=ang_vel_mag_true)
# p.camera.position = (60.0, 60.0, 20.0)
# p.show()

# # %%
# # Plotting orientation

# axes_ranges = [-1, 1, -1, 1, -1, 1]
# p = pv.Plotter(window_size=(1200, 1000))
# mrv.scatter3(
#     p,
#     sk,
#     point_size=10,
#     scalars=ang_err_deg,
#     clim=[ang_err_deg.min(), ang_err_deg.max()],
#     scalar_bar_args=dict(
#         title='$\mathbf{\sigma}$ error (deg)',
#         vertical=True,
#         position_x=0.85,
#         position_y=0.2,
#         height=0.5,
#         font_family='times',
#         title_font_size=30,
#         label_font_size=30,
#     ),
# )
# mrv.scatter3(p, s0, point_size=30, color='r', label='')
# p.show_grid(
#     xtitle='$\mathbf{\sigma_{1}(0)}$ [nondim]',
#     ytitle='$\mathbf{\sigma_{2}(0)}$ [nondim]',
#     ztitle='$\mathbf{\sigma_{3}(0)}$ [nondim]',
#     font_family='times',
#     font_size=12,
#     axes_ranges=axes_ranges,
# )
# p.camera.position = 0.8 * np.array([6.0, 6.0, 2.0])
# p.show()
