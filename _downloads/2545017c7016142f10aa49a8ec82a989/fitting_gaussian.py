"""
Fitting Gaussians
=================

Let's fit a Gaussian to some scattered data
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import mirage as mr


def plot_gaussian_fit(rv):
    mean = rv.mean
    mins = mean - 4 * np.sqrt(np.diag(rv.cov))
    maxs = mean + 4 * np.sqrt(np.diag(rv.cov))
    xpts = np.linspace(mins[0], maxs[0], 100)
    ypts = np.linspace(mins[1], maxs[1], 100)
    xx, yy = np.meshgrid(xpts, ypts)
    prob = rv.pdf(np.vstack((xx.flatten(), yy.flatten())).T).reshape(xx.shape)

    plt.contour(xpts, ypts, prob)
    plt.colorbar()

data = np.loadtxt(os.path.join(os.environ['SRCDIR'], '..', 'testfitdata.txt'))

rv_mle, inds = mr.fit_2d_gaussian(data, return_used_inds=True)

plt.figure()
plt.scatter(data[:, 0], data[:, 1], s=10, c=inds, cmap='cividis')
plt.scatter(rv_mle.mean[0], rv_mle.mean[1], marker='x', s=30, c='m')
plt.axis("equal")
plot_gaussian_fit(rv_mle)
plt.show()