"""
Rigid Body Torque Free Attitudes
================================

Animates the attitude motion of an object in torque free motion
"""

import numpy as np

import mirage as mr
import mirage.vis as mrv

(dates, epsecs) = mr.date_linspace(
    mr.now(), mr.now() + mr.seconds(10), 200, return_epsecs=True
)
obj = mr.SpaceObject('hylas4.obj', identifier=36411)
obj.build_pyvista_mesh()
itensor = np.diag((1.0, 2.0, 3.0))
q0 = np.array([0, 0, 0, 1])
wmag = 3  # rad/s

# %%
# Unstable rotation about the intermediate axis
obj_attitude = mr.RbtfAttitude(
    w0=wmag * mr.hat(np.array([0.1, 3, 0.1])),
    q0=q0,
    itensor=itensor,
)
(q, w) = obj_attitude.propagate(epsecs)
mrv.vis_attitude_motion(
    obj, q, 'tess_unstable.gif', framerate=20, background_color='white'
)

# %%
# Black version of unstable motion
mrv.vis_attitude_motion(
    obj, q, 'tess_unstable_black.gif', framerate=20, background_color='black'
)

# %%
# Spin and precession
obj_attitude = mr.RbtfAttitude(
    w0=wmag * mr.hat(np.array([0.1, 3, 0.1])),
    q0=q0,
    itensor=np.diag((1.0, 2.0, 2.0)),
)
(q, w) = obj_attitude.propagate(epsecs)
mrv.vis_attitude_motion(obj, q, 'tess_sp.gif', framerate=20, background_color='white')

# %%
# Stable rotation about the first axis
obj_attitude = mr.RbtfAttitude(
    w0=wmag * mr.hat(np.array([1.0, 0.0, 0.0])),
    q0=q0,
    itensor=itensor,
)
(q, w) = obj_attitude.propagate(epsecs)
mrv.vis_attitude_motion(
    obj, q, 'tess_stable.gif', framerate=20, background_color='white'
)
