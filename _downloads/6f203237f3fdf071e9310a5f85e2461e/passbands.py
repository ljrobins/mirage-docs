"""
Passbands
=========

Passbands for the Gaia and Johnson-Cousins photometric systems.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import mirage as mr
import mirage.vis as mrv


def plot_passband(lambdas, vals, label, color):
    ax = plt.gca()
    ax = sns.lineplot(x=lambdas, y=vals, color=color, alpha=0.5, label=label)
    ax.fill_between(lambdas, vals, color=color, alpha=0.1)


lambdas = np.linspace(300, 1200, 1000)
g_pass = mr.gaia_passband(lambdas, 'G')
g_bp_pass = mr.gaia_passband(lambdas, 'G_BP')
g_rp_pass = mr.gaia_passband(lambdas, 'G_RP')

ccd = mr.ChargeCoupledDevice(preset='pogs')
qe = ccd.quantum_efficiency(lambdas)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plot_passband(lambdas, g_pass, label='$G$', color='g')
plot_passband(lambdas, g_bp_pass, label='$G_{BP}$', color='b')
plot_passband(lambdas, g_rp_pass, label='$G_{RP}$', color='r')
# plot_passband(lambdas, qe, label="QE", color="k")
mrv.texit('Gaia Passbands', 'Wavelength [nm]', 'Transmission', grid=True)
plt.legend()

plt.subplot(1, 2, 2)
plot_passband(
    lambdas, mr.johnson_cousins_passbands(lambdas, 'U'), label='$U$', color='violet'
)
plot_passband(
    lambdas, mr.johnson_cousins_passbands(lambdas, 'B'), label='$B$', color='c'
)
plot_passband(
    lambdas, mr.johnson_cousins_passbands(lambdas, 'V'), label='$V$', color='g'
)
plot_passband(
    lambdas, mr.johnson_cousins_passbands(lambdas, 'R'), label='$R$', color='r'
)
plot_passband(
    lambdas, mr.johnson_cousins_passbands(lambdas, 'I'), label='$I$', color='maroon'
)

mrv.texit('Johnson-Cousins Passbands', 'Wavelength [nm]', 'Transmission', grid=True)
plt.legend()

plt.tight_layout()
plt.show()
