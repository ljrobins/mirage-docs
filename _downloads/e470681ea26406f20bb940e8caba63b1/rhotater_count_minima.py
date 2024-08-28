"""
Attitude Inversion Minima
=========================

"""

import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import mirage as mr
import pyvista as pv
import mirage.vis as mrv

x = pl.read_parquet('/Users/liamrobinson/Documents/rhotater/saved.parquet')

x = x.sort('fun')
x = x.with_columns(jmag=np.linalg.norm(x['gradient'].to_numpy(), axis=1))

xk = x['xk'].to_numpy()
x0 = x['x0'].to_numpy()
fun = x['fun'].to_numpy()


def compute_lc_from_x0(x0):
    obj = mr.SpaceObject('cube.obj')

    L = np.array([1.0, 0.0, 0.0])
    O = mr.hat(np.array([1.0, 1.0, 0.0]))
    brdf = mr.Brdf('phong', cd=0.5, cs=0.5, n=3)
    if x0.size == 7:
        itensor = np.abs(np.diag([1.0, 2.0, x0[-1]]))
    if x0.size == 8:
        itensor = np.abs(np.diag([1.0, x0[-2], x0[-1]]))
    else:
        itensor = np.diag([1.0, 2.0, 3.0])

    q0 = mr.mrp_to_quat(x0[:3].reshape(1, -1))
    teval = np.linspace(0, 6, 20)
    w0 = x0[3:6] / teval.max()
    q_of_t, w_of_t = mr.propagate_attitude_torque_free(q0, w0, itensor, teval)
    c_of_t = mr.quat_to_dcm(q_of_t)
    s_of_t = mr.quat_to_mrp(q_of_t)

    svb = mr.stack_mat_mult_vec(c_of_t, L)
    ovb = mr.stack_mat_mult_vec(c_of_t, O)
    lc = obj.convex_light_curve(brdf, svb, ovb)
    return lc / np.linalg.norm(lc)


x0i = xk[0]
x0t = np.array([1.0, 2.0, 3.0, 5.0, 4.0, 2.0, 1.0, 2.0])  # the truth
print(' x0i = ti.Vector([' + ', '.join([f'{x:6.3f}' for x in x0i]) + '])')
print(' x0t = ti.Vector([' + ', '.join([f'{x:6.3f}' for x in x0t]) + '])')

lc = compute_lc_from_x0(x0i)
lct = compute_lc_from_x0(x0t)


plt.plot(lct, linewidth=3, c='k', label='Observed')
plt.plot(lc, 'c--', linewidth=2, label='Converged')


lcs = []
for i in range(100):
    lcs.append(compute_lc_from_x0(xk[i]))

lc_std = np.array(lcs).std(axis=0)
plt.fill_between(
    np.arange(lc.size),
    lc - 3 * lc_std,
    lc + 3 * lc_std,
    color='r',
    alpha=0.3,
    label=rf'$\pm3\sigma$ for top {len(lcs)}',
)


mrv.texit(
    'Attitude inversion results - uniform cube',
    'Epoch seconds',
    'Normalized brightness',
)
plt.legend()
plt.show()

# %%
# Now let's plot a histogram of solution loss function values

s = xk[:, :3]
s = mr.quat_to_mrp(mr.quat_upper_hemisphere(mr.mrp_to_quat(s)))
w = xk[:, 3:6].copy()
w[mr.vecnorm(w).flatten() > 4.0, :] = np.nan

lf = np.log10(fun)
loss_frac = (lf - lf.max()) / (lf.min() - lf.max())

print(loss_frac)

plt.hist(fun, bins=np.geomspace(1e-4, 1e1, 100))
plt.xscale('log')
plt.yscale('log')
plt.title('Solutions')
plt.xlabel('Loss values')
plt.ylabel('Count')
plt.grid()
plt.show()


# %%
# And the distribution of solutions in MRP space

p = pv.Plotter()
p.set_background('k')
mrv.scatter3(
    p,
    s,
    scalars=loss_frac,
    opacity=loss_frac,
    cmap='hot',
    point_size=5,
    show_scalar_bar=False,
)
# p.show()
mrv.orbit_plotter(p, gif_name='s_sols.gif', focus=(0.0, 0, 0))

# %%
# And the distribution of solutions in angular velocity space

p = pv.Plotter()
p.set_background('k')
mrv.scatter3(
    p,
    w,
    scalars=loss_frac,
    opacity=loss_frac,
    cmap='cool',
    point_size=5,
    show_scalar_bar=False,
)
# p.show()
mrv.orbit_plotter(p, gif_name='w_sols.gif', focus=(0, 0, 0))
