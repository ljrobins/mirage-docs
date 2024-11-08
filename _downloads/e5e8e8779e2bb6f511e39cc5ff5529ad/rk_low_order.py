"""
Comparing Runge-Kutta Methods
=============================

How simple can a Runge-Kutta method get to still yield "good" performance for attitude propagation?

"""

import time

import numpy as np

import mirage as mr


def omega_dot(w: np.ndarray, itensor: np.ndarray) -> np.ndarray:
    wdot = np.array([0.0, 0.0, 0.0])
    wdot[0] = -1 / itensor[0] * (itensor[2] - itensor[1]) * w[1] * w[2]
    wdot[1] = -1 / itensor[1] * (itensor[0] - itensor[2]) * w[2] * w[0]
    wdot[2] = -1 / itensor[2] * (itensor[1] - itensor[0]) * w[0] * w[1]
    return wdot


def mrp_kde(s: np.ndarray, w: np.ndarray) -> np.ndarray:
    s1, s2, s3 = s[0], s[1], s[2]
    s12 = s1**2
    s22 = s2**2
    s32 = s3**2

    m = (
        1
        / 4
        * np.array(
            [
                [1 + s12 - s22 - s32, 2 * (s1 * s2 - s3), 2 * (s1 * s3 + s2)],
                [2 * (s1 * s2 + s3), 1 - s12 + s22 - s32, 2 * (s2 * s3 - s1)],
                [2 * (s1 * s3 - s2), 2 * (s2 * s3 + s1), 1 - s12 - s22 + s32],
            ]
        )
    )
    return m @ w


def tilde(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def state_derivative_mrp(x, itensor):
    xdot = 0 * x
    s = x[:3]
    w = x[3:6]
    xdot[:3] = mrp_kde(s, w)
    xdot[3:6] = omega_dot(w, itensor)
    return xdot


def rk4_step(x0, h, itensor):
    k1 = h * state_derivative_mrp(x0, itensor)
    k2 = h * state_derivative_mrp(x0 + k1 / 2, itensor)
    k3 = h * state_derivative_mrp(x0 + k2 / 2, itensor)
    k4 = h * state_derivative_mrp(x0 + k3, itensor)
    return x0 + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6


def rk2_step(x0, h, itensor):
    k1 = h * state_derivative_mrp(x0, itensor)
    k2 = h * state_derivative_mrp(x0 + k1 / 2, itensor)
    return x0 + k2


def rk3_step(x0, h, itensor):
    k1 = h * state_derivative_mrp(x0, itensor)
    k2 = h * state_derivative_mrp(x0 + k1 / 2, itensor)
    k3 = h * state_derivative_mrp(x0 + 2 * k2 - k1, itensor)
    return x0 + 1 / 6 * (k1 + 4 * k2 + k3)


def integrate(x0: np.ndarray, h: float, intensor: np.ndarray, ntimes: int, method: str):
    x0 = x0.copy()

    t1 = time.time()
    for _ in range(ntimes):
        if method.lower() == 'rk2':
            x0 = rk2_step(x0, h, itensor)
        if method.lower() == 'rk3':
            x0 = rk3_step(x0, h, itensor)
        if method.lower() == 'rk4':
            x0 = rk4_step(x0, h, itensor)

    return x0, time.time() - t1


if __name__ == '__main__':
    h = 0.1
    ntimes = 100
    teval = np.linspace(0, h * ntimes, ntimes)

    itensor = np.array([1.0, 2.0, 3.0])
    s0 = np.array([1.0, 1.0, 1.0])
    w0 = np.array([1.0, 4.0, 2.0])
    q0 = mr.mrp_to_quat(s0)
    x0 = np.concatenate((s0, w0))

    xf, dt2 = integrate(x0, h, itensor, ntimes, method='rk2')
    qf_rk2 = mr.mrp_to_quat(xf[:3])

    xf, dt3 = integrate(x0, h, itensor, ntimes, method='rk3')
    qf_rk3 = mr.mrp_to_quat(xf[:3])

    xf, dt4 = integrate(x0, h, itensor, ntimes, method='rk4')
    qf_rk4 = mr.mrp_to_quat(xf[:3])

    mr.tic('RK45 time: ')
    q_of_t, w_of_t = mr.integrate_rigid_attitude_dynamics(
        q0, w0, np.diag(itensor), teval
    )
    mr.toc()
    qf_rk45 = q_of_t[-1, :]

    ang_deg = np.rad2deg(mr.quat_ang(qf_rk45, qf_rk2))
    print(f'RK2 is {ang_deg:.2e} deg from RK45 in {dt2:.1e} sec')

    ang_deg = np.rad2deg(mr.quat_ang(qf_rk45, qf_rk3))
    print(f'RK3 is {ang_deg:.2e} deg from RK45 in {dt3:.1e} sec')

    ang_deg = np.rad2deg(mr.quat_ang(qf_rk45, qf_rk4))
    print(f'RK4 is {ang_deg:.2e} deg from RK45 in {dt4:.1e} sec')

    obj = mr.SpaceObject('tess.obj')
    # mrv.vis_attitude_motion(obj, q_of_t, framerate=30)
