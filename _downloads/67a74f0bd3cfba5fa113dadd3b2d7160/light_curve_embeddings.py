"""
Brightness Function Embeddings
==============================

Experimenting with different ways to embed the brightness function of a given shape into a lower-dimensional space.
"""

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import mirage as mr

# %% Defining the object and its brightness function

obj = mr.SpaceObject("cube.obj")
brdf = mr.Brdf("phong", cd=0.5, cs=0.5, n=5.0)
bf = lambda svb, ovb: obj.convex_light_curve(brdf, svb, ovb)

# %% Calculating the brightness function at a bunch of points

n = 10_000
svb = mr.rand_unit_vectors(n)
ovb = mr.rand_unit_vectors(n)

X = np.hstack((svb, ovb))
Y = bf(svb, ovb)

import sklearn.manifold as manifold

mr.tic()
iso = manifold.TSNE(n_components=2)
# iso.fit(X, Y)
X_iso = iso.fit_transform(X)
mr.toc()

# %% Plotting the brightness function in the embedding space

plt.scatter(X_iso[:, 0], X_iso[:, 1], c=Y)
plt.colorbar()
plt.show()