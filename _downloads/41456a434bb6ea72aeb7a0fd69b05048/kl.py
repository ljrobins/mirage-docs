"""
KL Divergence
=============
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

x = np.linspace(-5, 5, 10000)
p = norm(scale=0.5, loc=0.0).pdf(x)
q = norm(scale=0.1, loc=2).pdf(x)


def kl_divergence(p: np.ndarray, q: np.ndarray):
    p /= p.sum()
    q /= q.sum()
    valid = (p / q) > 1e-1
    return np.sum(p[valid] * np.log(p[valid] / q[valid]))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray):
    m = (p + q) / 2
    return 1 / 2 * (kl_divergence(p, m) + kl_divergence(q, m))


def jsd_norm(p: np.ndarray, q: np.ndarray):
    from scipy.spatial.distance import jensenshannon

    return 1.0 / np.log(2) * jensenshannon(p, q) ** 2


print(f'Surprisal from approximating P by Q: {kl_divergence(p, q):.2f}')
print(f'Surprisal from approximating Q by P: {kl_divergence(q, p):.2f}')

print(
    f'scaled JSD of Q and P: {2 / np.sqrt(np.e) * jensen_shannon_divergence(q, p):.4f} (scipy: {jsd_norm(p, q):.4f})'
)  # max

plt.plot(x, p, label='$P(x)$')
plt.plot(x, q, label='$Q(x)$')
plt.grid()
plt.xlabel('$x$')
plt.ylabel('Probability density')
plt.legend()
plt.tight_layout()
plt.show()
