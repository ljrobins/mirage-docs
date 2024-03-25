"""
Precise Orbit Determination with Batch Least Squares
====================================================
"""

import datetime

import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv

# %%
# First, let's define a station and a truth object

station = mr.Station()
obj = mr.SpaceObject("cube.obj", identifier=36411)

# %%
# Let's create three line of sight observations of the object

dates_iod = mr.date_linspace(mr.now(), mr.now() + mr.minutes(1), 3)
lhats_iod = station.object_line_of_sight(obj, dates_iod, apparent=False)
ras_rad, decs_rad = mr.eci_to_ra_dec(lhats_iod)
rv2 = mr.gauss_iod(station, dates_iod, ras_rad, decs_rad)

# %%
# Now we collect a lot more observations

dates_pod = mr.date_arange(mr.now(), mr.now() + mr.hours(1), mr.seconds(20))
lhats_pod = station.object_line_of_sight(obj, dates_pod, apparent=False)
ras_rad, decs_rad = mr.eci_to_ra_dec(lhats_pod)

# %%
# Let's compute the H matrix for this observation set. We know that the H matrix is defined as the partial derivatives of the observations with respect to the state vector.


def angles_only_measurement_model(state: np.ndarray, dates: np.ndarray) -> np.ndarray:
    assert (
        (state.shape[0] == len(dates))
        if isinstance(dates, np.ndarray)
        else state.size == 6
    ), "state and dates must have the same number of rows"
    r = state.reshape(-1, 6)[:, :3]
    r_station = station.j2000_at_dates(dates)
    r_station_to_sat = r - r_station
    return np.concatenate(mr.eci_to_ra_dec(r_station_to_sat)).reshape(-1, 2)


def angles_only_measurement_model_jacobian(
    state: np.ndarray, dates: np.ndarray
) -> np.ndarray:
    assert (
        (state.shape[0] == len(dates))
        if isinstance(dates, np.ndarray)
        else state.size == 6
    ), "state and dates must have the same number of rows"
    r = state.reshape(-1, 6)[:, :3]
    r_station = station.j2000_at_dates(dates)
    r_station_to_sat = r - r_station
    wsq = mr.vecnorm(r_station_to_sat[:, :2]).flatten() ** 2
    w = np.sqrt(wsq)
    rho_sq = mr.vecnorm(r_station_to_sat).flatten() ** 2

    zero = np.zeros_like(wsq)
    H = np.array(
        [
            [-r[:, 1] / wsq, r[:, 0] / wsq, zero, zero, zero, zero],
            [
                -r[:, 0] * r[:, 2] / (w * rho_sq),
                -r[:, 1] * r[:, 2] / (w * rho_sq),
                w / rho_sq,
                zero,
                zero,
                zero,
            ],
        ]
    ).squeeze()

    H2 = H.transpose(2, 0, 1)
    H2 = H2.reshape(-1, 6)
    return H2


from typing import Callable


def lumve(
    measurements: np.ndarray,
    dates: np.ndarray[datetime.datetime],
    measurement_model: Callable,
    measurement_model_jacobian: Callable,
    initial_state: np.ndarray,
    initial_covariance: np.ndarray,
) -> np.ndarray:
    xk = initial_state
    Pk = initial_covariance
    for _ in range(10):
        states = mr.integrate_orbit_dynamics(xk, dates)
        H = measurement_model_jacobian(states, dates)
        K = Pk @ H.T @ np.linalg.inv(H @ Pk @ H.T)
        Pk = (np.eye(6) - K @ H) @ Pk
        # xk += K @ (measurements - measurement_model(states, dates))
        xk += (
            np.linalg.inv(H.T @ H)
            @ H.T
            @ (measurements - measurement_model(states, dates))
        )
    return xk, Pk


# %%
# Now let's run the batch least squares algorithm

measurements = np.vstack(mr.eci_to_ra_dec(lhats_pod)).T
initial_covariance = np.eye(6) * 1e-6
initial_state = rv2
# xk, Pk = lumve(
#     measurements,
#     dates_pod,
#     angles_only_measurement_model,
#     angles_only_measurement_model_jacobian,
#     initial_state,
#     initial_covariance,
# )
