"""
Relative orbit sequence
========================
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr

coe_chief = np.array([7000, 0.1, 10, 5, 5, 5])
coe_chase = coe_chief + np.array([0, 0.0005, 0.0005, 0.01, 0.01, 0.0])
dates = mr.date_linspace(mr.now(), mr.now() + mr.hours(1.5), 100)
rv0_chief = mr.coe_to_rv(coe_chief)
rv0_chase = mr.coe_to_rv(coe_chase)

rv_chase = mr.integrate_orbit_dynamics(rv0_chief, dates)
r_chase = rv_chase[:, :3]
rv_chief = mr.integrate_orbit_dynamics(rv0_chase, dates)
r_chief = rv_chief[:, :3]

obj_chief = mr.SpaceObject('matlib_hylas4.obj', identifier='goes 15')
sv = mr.sun(dates)
nadir = -mr.hat(r_chief)
att_chief = mr.AlignedAndConstrainedAttitude(
    v_align=nadir, v_const=sv, dates=dates, axis_order=(1, 2, 0)
)
svi = mr.hat(mr.sun(dates))
dcm_inertial_to_chief = att_chief.dcms_at_dates(dates)
q_chief = mr.dcm_to_quat(dcm_inertial_to_chief)
svb = mr.stack_mat_mult_vec(dcm_inertial_to_chief, svi)

camera_z = mr.hat(r_chief - r_chase)
ref = mr.hat(np.cross(r_chief, rv_chief[:, 3:]))
camera_y = mr.hat(np.cross(camera_z, ref))
camera_x = np.cross(camera_y, camera_z)
dcm_inertial_to_camera = np.swapaxes(
    np.swapaxes(np.stack((camera_x, camera_y, camera_z), axis=-1), 0, 2), 0, 1
)
q_camera = mr.dcm_to_quat(dcm_inertial_to_camera)

# mrv.vis_attitude_motion(obj_chief, mr.dcm_to_quat(att_chief.dcms_at_dates(dates)), fname='test.gif')


# %%
# Running the synthetic image generator

import mirage.synth as mrsyn

mrsyn.generate_synthetic_sequence(
    'matlib_goes17.obj',
    svb,
    camera_pos=r_chase - r_chief,
    q_camera=q_camera,
    q_chief=q_chief,
    fov_deg=37,
)

# %%
# Making a matplotlib animation of the result

from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
ax.set_aspect('equal')
fig.patch.set_facecolor('black')

ims = []
for imname in sorted(
    [x for x in os.listdir('out') if '.png' in x], key=lambda x: int(x[5:-4])
):
    ims.append(plt.imread(f'out/{imname}'))

im = plt.imshow(ims[0][:, :, 0], cmap='gray')


def animate(i):
    im.set_data(ims[i][:, :, 0])
    return (im,)


frames = len(dates)
anim_time = 10
fps = frames / anim_time
interval = 1000 / fps
anim = FuncAnimation(fig, animate, frames=frames, interval=interval, blit=True)
anim.save('rpo.mp4')
plt.show()
