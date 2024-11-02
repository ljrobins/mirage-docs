"""
Self-Shadowing Methods
======================

Comparing light curves produced by different shadowing methods
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import mirage as mr

mr.set_model_directory(
    '/Users/liamrobinson/Documents/maintained-research/mirage-models/Non-Convex/'
)
obj = mr.SpaceObject('irregular.obj')

brdf = mr.Brdf('blinn-phong', cd=0.5, cs=0.5, n=5.0)
df = pl.read_parquet(os.path.join(os.environ['SRCDIR'], '..', 'saved.parquet'))
print(df)
x0 = np.array([0.0, 0.0, 0.0, -3.0, 3.0, 1.0])
lc = [
    0.341383,
    0.197364,
    0.123992,
    0.118558,
    0.144448,
    0.199170,
    0.281262,
    0.420311,
    0.634635,
    0.868985,
    1.099806,
    1.296503,
    1.425755,
    1.462219,
    1.400128,
    1.286364,
    1.145995,
    1.006063,
    0.886744,
    0.765302,
    0.624586,
    0.485581,
    0.408199,
    0.407272,
    0.409617,
    0.467555,
    0.470402,
    0.466237,
    0.426641,
    0.379301,
]

itensor = np.diag([1.0, 2.0, 3.0])
t = np.linspace(0, 1, len(lc), endpoint=False)
svi = np.tile(np.array([1.0, 0.0, 0.0]), (len(t), 1))
ovi = np.tile(np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0]), (len(t), 1))
# ovi = np.array([np.cos(t), np.sin(t), 0.0*t]).T

s0 = x0[:3]
w0 = x0[3:]

attitude = mr.RbtfAttitude(w0, mr.mrp_to_quat(s0), itensor)
q_of_t, _ = attitude.propagate(t)
c_of_t = mr.quat_to_dcm(q_of_t)
svb = mr.stack_mat_mult_vec(c_of_t, svi)
ovb = mr.stack_mat_mult_vec(c_of_t, ovi)

lc_engine = mr.run_light_curve_engine(
    brdf, obj, svb, ovb, show_window=True, instances=1, frame_rate=40
)
lc_convex = obj.convex_light_curve(brdf, svb, ovb)

plt.plot(lc_engine, 'k', linewidth=3, label='Rendered shadows')
plt.plot(lc_convex, linewidth=2.5, label='Convex')
plt.plot(lc, 'r--', linewidth=2.5, label='Analytic shadows')
plt.grid()
plt.xlabel('Timestep')
plt.ylabel('Normalized irradiance')
plt.legend()
plt.show()
