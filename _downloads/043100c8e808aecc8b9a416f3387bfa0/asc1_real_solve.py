"""
Solving for NOAA 13 SDLR
========================

"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from image_similarity_measures.quality_metrics import psnr
from scipy.optimize import shgo

import mirage as mr

np.random.seed(1)

# print(ssim(org_img=np.random.rand(100,100,1), pred_img=np.random.rand(100,100,1)))
# enddd


def evaluate_total_pdf(
    pts: np.ndarray, ns: np.ndarray, d_eval: np.ndarray
) -> np.ndarray:
    """
    Evaluates the total weighted PDF at depths d_eval.

    :param pts: Triangles endpoints [N, 3, 2] (depths in col 0, probs in col 1)
    :param ns: Weights for each triangle [N,]
    :param d_eval: Depths to evaluate at [M,]
    :return: Total PDF values [M,]
    """
    # Reshape for broadcasting:
    # d_eval: (1, M)
    # pts components: (N, 1)
    d = d_eval[np.newaxis, :]
    d0 = pts[:, 0, 0][:, np.newaxis]
    d1 = pts[:, 1, 0][:, np.newaxis]
    d2 = pts[:, 2, 0][:, np.newaxis]
    p1 = pts[:, 1, 1][:, np.newaxis]
    w = ns[:, np.newaxis]

    # Masking for the two linear "legs" of the triangular distributions
    mask1 = (d >= d0) & (d < d1)
    mask2 = (d >= d1) & (d <= d2)

    # Linear interpolation: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    # For Leg 1: p = 0 + (d - d0) * (p1 - 0) / (d1 - d0)
    # We use np.where to avoid division by zero on degenerate triangles
    denom1 = np.where(d1 != d0, d1 - d0, 1.0)
    vals1 = mask1 * (d - d0) * (p1 / denom1)

    # For Leg 2: p = p1 + (d - d1) * (0 - p1) / (d2 - d1)
    denom2 = np.where(d2 != d1, d2 - d1, 1.0)
    vals2 = mask2 * p1 * (1 - (d - d1) / denom2)
    face_pdfs = vals1 + vals2
    total_pdf = np.sum(w * face_pdfs, axis=0)

    return total_pdf


station = mr.Station('izana')

norad = 22739
df = (
    pl.scan_parquet(
        '/Users/liamrobinson/Documents/maintained-research/mirage/scratch/digos/data/proc/digos-dlr.parquet'
    )
    .filter(pl.col.norad == norad)
    .collect()
).with_columns(epsecs=(pl.col.date - pl.col.date.min()).dt.total_nanoseconds() / 1e9)

obj, _ = mr.parametric_scyl_wing(
    body_length=3.71,
    body_radius=1.88 / 2,
    panel_length=4.91,
    panel_width=2.37,
    panel_angle_rad=np.deg2rad(20),
)

laser = mr.Laser(
    hz=400,
    lam=1064,
    pulse_energy=200e-3,
    beam_divergence_half_angle=mr.dms_to_rad(0, 0, 15),
    tracking_jitter_rad=mr.dms_to_rad(0, 0, 1),
    beam_pointing_error_rad=mr.dms_to_rad(0, 0, 0),
    pulse_fwhm_ns=7,
)


xbins = np.linspace(df['epsecs'].min(), df['epsecs'].max(), 100)
xbc = xbins[:-1] + xbins[1] / 2 - xbins[0] / 2
ybins = np.linspace(df['est_depth_poly'].min() - 1, df['est_depth_poly'].max() + 1, 50)
ybc = ybins[:-1] + ybins[1] / 2 - ybins[0] / 2
dates_linear = np.array([df['date'][0] + mr.seconds(s) for s in xbins])
obj_prop = mr.SpaceObject('cube.obj', identifier=22739)

r_obj = obj_prop.propagate(dates_linear)
r_station = station.j2000_at_dates(dates_linear)
ovi = mr.hat(r_station - r_obj)

wmag_min = np.deg2rad(180 / (44 - 25))
wmag_max = np.deg2rad(360 / (44 - 25))

observed_img, _, _ = np.histogram2d(
    df['epsecs'], df['est_depth_poly'], bins=(xbins, ybins)
)
_sum = observed_img.sum(axis=1, keepdims=True)
keep_times_inds = _sum.flatten() > 20
keep_times = xbc[keep_times_inds]

_sum[_sum == 0] = 1.0
observed_img_time_norm = observed_img / _sum
observed_img_time_norm = observed_img_time_norm[keep_times_inds, :]

observed_img_all_norm = observed_img / observed_img.sum()
observed_img_all_norm = observed_img_all_norm[keep_times_inds, :]


def wmag_sampler(num: int, sigma_deg: float = 0.5):
    r = np.random.rand(num)
    w = np.zeros_like(r)
    w[r > 0.5] = np.random.normal(
        loc=wmag_min, scale=np.deg2rad(sigma_deg), size=(r > 0.5).sum()
    )
    w[r <= 0.5] = np.random.normal(
        loc=wmag_max, scale=np.deg2rad(sigma_deg), size=(r <= 0.5).sum()
    )
    return w


def img_dist_wmse(A: np.ndarray, B: np.ndarray) -> float:
    """
    Weighted Mean Squared Error: Penalizes mismatches more
    heavily in regions where Image A is bright.
    """
    weights = A / np.max(A) if np.max(A) > 0 else A
    return np.mean(weights * (A - B) ** 2)


def img_dist_ncc(A: np.ndarray, B: np.ndarray) -> float:
    """
    Normalized Cross-Correlation: Measures spatial alignment.
    Returns a value between -1 and 1 (1 being perfect correlation).
    """
    a_mean = A - np.mean(A)
    b_mean = B - np.mean(B)
    numerator = np.sum(a_mean * b_mean)
    denominator = np.sqrt(np.sum(a_mean**2) * np.sum(b_mean**2))
    return numerator / denominator if denominator > 0 else 0.0


def img_dist_asymmetric(A: np.ndarray, B: np.ndarray) -> float:
    """
    Asymmetric Penalty: Only penalizes pixels where Image A
    is brighter than Image B. Extra signal in B is free.
    """
    diff = A - B
    # Square the missing signal to emphasize larger gaps
    penalty = np.where(diff > 0, diff**2, 0)
    return np.mean(penalty)


def img_dist_soft_recall(A: np.ndarray, B: np.ndarray) -> float:
    """
    Soft Recall: Ratio of signal in A 'covered' by B.
    1.0 means B covers all of A's signal intensity.
    """
    total_signal_a = np.sum(A)
    if total_signal_a == 0:
        return 1.0
    covered_signal = np.sum(np.minimum(A, B))
    return covered_signal / total_signal_a


def img_dist_simple(
    A: np.ndarray, B: np.ndarray
):  # A: observed image, may have gaps, B: estimated image
    # return np.sum(np.max(np.dstack((np.zeros_like(A), A-B)), axis=-1))
    A = A.copy()
    B = B.copy()
    # A[A < np.percentile(A, 50)] = 0.0
    # B[B < np.percentile(B, 50)] = 0.0
    d = np.sum(np.abs(A - B))
    print(d)
    return d


# fsim, issm, psnr, rmse, sam, sre, ssim, uiq
def img_dist(A, B):
    d = -psnr(A[:, :, None], B[:, :, None])
    print(d)
    return d


# n = int(1e3)
# wmags = wmag_sampler(n)
# quats = mr.rand_quaternions(n)
# axes = np.random.randint(0, 2, n)
# data = []

bounds = np.vstack(
    (
        3 * [[-1, 1]],
        [[wmag_max - 0.01, wmag_max + 0.01]],
    )
)


def pds_of_x(x):
    q = mr.mrp_to_quat(x[:3])
    omega0 = np.array([0, x[3], 0])
    brdf = mr.Brdf('blinn-phong', cd=0.5, cs=0.5, n=10)

    q_of_t, _ = mr.propagate_attitude_torque_free(
        q, omega0, np.diag([1, 1, 1]), teval=keep_times
    )

    ovb = mr.stack_mat_mult_vec(mr.quat_to_dcm(q_of_t), ovi[:-1][keep_times_inds])
    sigma = brdf.compute_reflection_matrix(ovb, ovb, obj.face_normals) * obj.face_areas
    ns = mr.laser_link(
        r_obj[:-1][keep_times_inds],
        r_station[:-1][keep_times_inds],
        sigma,
        atmospheric_transmission=1.0,
        pulse_energy=laser.pulse_energy,
        transmit_wavelength=laser.lam,
        beam_divergence_half_angle_rad=laser.beam_divergence_half_angle,  # half the full beam width divergence mean
        tracking_jitter_rad=laser.tracking_jitter_rad,
        beam_pointing_error_rad=laser.beam_pointing_error_rad,
        receiver_area=station.telescope.aperture_area,
        transmit_optical_efficiency=1.0,
        receive_optical_efficiency=1.0,
    )

    pds = []
    for _o, _n in zip(ovb, ns):
        pts = mr.triangle_depth_pdf_vectorized(obj.fv, -_o)
        pd = evaluate_total_pdf(pts, _n, ybc)
        pds.append(pd)
    pds = np.array(pds)
    pds_time_norm = pds / pds.sum(axis=1, keepdims=True)
    pds_all_norm = pds / pds.sum()
    y_bin_mean = np.sum(np.arange(ybc.size)[None, :] * pds_time_norm, axis=1)
    for i in range(pds.shape[0]):
        pds_all_norm[i] = np.roll(pds_all_norm[i], -int(y_bin_mean[i] - ybc.size // 2))
    return pds_all_norm


def f(x: np.ndarray) -> float:
    pds_all_norm = pds_of_x(x)
    # print(y_bin_mean)
    # print(y_bin_mean.shape)
    # endd

    # plt.subplot(1,2,1)
    # plt.imshow(pds_all_norm.T)
    # plt.colorbar()
    # plt.subplot(1,2,2)
    # plt.imshow(observed_img_all_norm.T)
    # plt.colorbar()
    # plt.show()

    d = img_dist(observed_img_all_norm, pds_all_norm)
    return d


# sol, fsol = mr.cmaes(f, np.ones(6), population_size=100, max_generations=100)

sol = shgo(f, bounds=bounds, iters=2)
print(sol)
print(f'quat0=np.{repr(mr.mrp_to_quat(sol.x[:3]))}\nomega0mag={sol.x[3]}')

pds = pds_of_x(sol.x)


# with alive_bar(n) as bar:
#     for wmag, q, a in zip(wmags, quats, axes):
#         omega0 = np.zeros(3)
#         omega0[a] = wmag
#         bar()

#         data.append({'d': d, 'pds': pds})

# data.sort(key=lambda x: x['d'])


plt.subplot(1, 2, 1)
plt.imshow(pds.T)
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(observed_img_all_norm.T)
plt.gca().set_facecolor('k')
plt.colorbar()
plt.show()
