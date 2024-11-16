"""
Simple Box-Wing Light Curves
============================
"""

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

station = mr.Station()
obj = mr.SpaceObject('matlib_goes17.obj', identifier='GOES 17')

a_pan = 3.0 * 1.27  # m^2
a_bus = 0.8 * 1.1  # m^2
cd_pan = 0.5
cs_pan = 0.5
n_pan = 5
cd_bus = 0.5
cs_bus = 0.5
n_bus = 5

d0 = mr.utc(2024, 10, 18)
dates, epsecs = mr.date_linspace(d0, d0 + mr.hours(14), 100, return_epsecs=True)

r_obj = obj.propagate(dates)
svi = mr.sun(dates) - r_obj  # from obj to sun
ovi = station.j2000_at_dates(dates) - r_obj  # from obj to obs

f = (
    mr.AstroConstants.sun_irradiance_vacuum
    / mr.vecnorm(ovi) ** 2
    / (mr.vecnorm(svi) / mr.AstroConstants.au_to_km) ** 2
)
f = f.flatten()

attitude = mr.AlignedAndConstrainedAttitude(
    -mr.hat(r_obj), mr.hat(svi), dates, axis_order=(2, 0, 1)
)

q_of_t, w_of_t = attitude.propagate(epsecs)
d_of_t = mr.quat_to_dcm(q_of_t)  # inertial to body

svb = mr.stack_mat_mult_vec(d_of_t, svi)
ovb = mr.stack_mat_mult_vec(d_of_t, ovi)

s = mr.hat(svb)
o = mr.hat(ovb)
h = mr.hat(s + o)
fr_pan = cd_pan / np.pi + cs_pan * (n_pan + 2) / (2 * np.pi) * mr.dot(s, h) ** n_pan / (
    4 * mr.dot(s, o)
)
fr_bus = cd_bus / np.pi + cs_bus * (n_bus + 2) / (2 * np.pi) * mr.dot(o, h) ** n_bus / (
    4 * mr.dot(s, o)
)
lc_simp = ((a_bus * fr_bus + a_pan * fr_pan) * mr.rdot(s, o)).flatten()

brdf = mr.Brdf('blinn-phong', cd=0.5, cs=0.5, n=5)

lc_fancy = mr.run_light_curve_engine(
    brdf,
    obj,
    mr.hat(svb),
    mr.hat(ovb),
    show_window=True,
    verbose=False,
    rotate_panels=True,
    frame_rate=1000,
    instances=1,
)

# lc_sampler, info = station.observe_light_curve(obj, attitude, brdf, dates, integration_time_s=10, use_engine=True, rotate_panels=True)
# lc = lc_sampler().flatten()
# lc_fancy = info['lc_clean_norm'].flatten()

percent_err_at_peak = np.abs(lc_simp.max() - lc_fancy.max()) / lc_fancy.max() * 100
median_err = np.median(np.abs(lc_simp - lc_fancy))
print(percent_err_at_peak, median_err)

plt.plot(epsecs / 3600, lc_fancy * f, 'k', linewidth=2.5, label='Full Rendered')
plt.plot(epsecs / 3600, lc_simp * f, 'c--', linewidth=2, label='Simplified')
plt.grid()
plt.legend()
plt.title('Integrated Irradiance from GOES 17')
plt.xlabel('Hours after 00:00 Oct 18, 2024 UTC')
plt.ylabel('Integrated Light Curve [W / $m^2$]')
plt.show()

# %%
# Plotting the overall configuration

obj = mr.SpaceObject('box_wing.obj')
obj.shift_to_center_of_mass()
obj.v[np.abs(obj.v[:, 0]) > 1.0, 0] *= 2
obj = mr.SpaceObject(vertices_and_faces=(obj.v, obj.f))

s = mr.hat(np.array([0.0, 0.5, 0.4]))
nb = np.array([0.0, 1.0, 0.0])
ang = mr.angle_between_vecs(s, nb)
rotm = mr.r1(-ang)
faces_to_rotate = mr.vecnorm(obj.face_centroids) > 0.58
verts_to_rotate = np.unique(obj.f[faces_to_rotate.flatten()])
obj.v[verts_to_rotate, :] = mr.stack_mat_mult_vec(rotm, obj.v[verts_to_rotate, :])
obj = mr.SpaceObject(vertices_and_faces=(obj.v, obj.f))

p = pv.Plotter()
mrv.render_spaceobject(p, obj, scalars=obj.face_areas, cmap='blues')
mrv.plot_arrow(p, [0, 0, 0], [0, 1, 0], label='Nadir', color='green', scale=1.5)
mrv.plot_arrow(p, [0, 0, 0], s, label='Sun', color='yellow', scale=1.5)
mrv.plot_arrow(p, [0, 0, 0], [0, 1.0, 0.3], label='Observer', scale=1.5, color='Red')
p.show()
