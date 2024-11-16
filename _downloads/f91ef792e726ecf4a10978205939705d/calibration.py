"""
Subframes
=========

Displaying subframes of object signals, complete with their
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import mirage as mr
import mirage.photo as mrp
import mirage.vis as mrv


def calibration_df(
    impath: str, median_flat: np.ndarray, median_dark: np.ndarray
) -> pl.DataFrame:
    im_name = os.path.split(impath)[1]
    mr.tic(f'{im_name}: load fits')
    info = mr.info_from_fits(impath, telescope=station.telescope, minimal=True)
    mr.toc()

    light_with_flat_calibration = (
        (info['ccd_adu'] - median_dark) / median_flat
    ).astype(np.int32)

    mr.tic(f'{im_name}: background determination')
    image_sub, background_map, br_mask, background_std = mrp.background_subtract(
        light_with_flat_calibration, reduction_factor=2
    )
    mr.toc()

    # plt.imshow(np.log10(np.abs(image_sub)+1), cmap='gray')
    # plt.show()
    # endd

    mr.tic(f'{im_name}: object mask')
    obj_mask = mrp.object_mask(image_sub, background_std)
    mr.toc()

    mr.tic(f'{im_name}: compute contours')
    contour_df = mrp.compute_contours(obj_mask, reduction_factor=4)
    mr.toc()

    constraints = [
        dict(col='area', min=50, max=np.inf, include=True),
        dict(col='shape_factor', min=0.7, max=1, include=True),
    ]

    streak_contours = mrp.filter_contours(contour_df, constraints)

    mr.tic(f'{im_name}: astrometry')
    wcs = mrp.wcs_from_contours(info, streak_contours)
    look_dir_eci, up_dir_eci = mrp.look_and_up_from_wcs(station.telescope, wcs)
    uvs, spec = catalog.in_fov(look_dir_eci, up_dir_eci)

    centroids = streak_contours.select('x_center', 'y_center').to_numpy()

    radec = np.deg2rad(wcs.all_pix2world(centroids, 1))
    uvs_centroids = mr.ra_dec_to_eci(*radec.T)

    d, inds = catalog._tree.query(uvs_centroids, k=1)
    # pix_away = (
    #     mr.AstroConstants.rad_to_arcsecond * d / station.telescope.ccd.pixel_scale
    # )

    fluxes = []
    for centroid in centroids:
        cropped_adu = mrp.crop_at(image_sub, crop_location=centroid)
        flux, signal_mask = mrp.source_flux_from_subframe(
            cropped_adu, signal_mask_threshold=8
        )
        fluxes.append(flux)
        # plt.imshow(cropped_adu * signal_mask)
        # plt.show()

    f = (
        mr.integrated_spectrum(
            station,
            np.pi / 2 - info['center_elevation_rad'],
            spectrum=catalog._spec[inds.flatten()],
            lambdas=catalog._extra_vars['lambdas'],
        )
        * info['integration_time']
    )

    df = pl.DataFrame(
        {
            'predicted_flux': f.flatten(),
            'observed_flux': fluxes,
            'catalog_ind': inds.flatten(),
        }
    ).drop_nulls()
    print(df)

    df = df.with_columns(file_path=[impath] * df.height)
    return df.unique(subset='catalog_ind', keep='none')


station = mr.Station()


def calibrate_flat(flats_dir: str) -> np.ndarray:
    flats_paths = [os.path.join(flats_dir, f) for f in os.listdir(flats_dir)[:10]]

    # info = mr.info_from_fits(flats_paths[0], telescope=station.telescope, minimal=True)
    # flat = info['ccd_adu']
    # flat_sub, background_map, br_mask, background_std = mrp.background_subtract(
    #     flat, reduction_factor=2
    # )

    # win_mean = ndimage.uniform_filter(flat_sub, (20,20))
    # win_sqr_mean = ndimage.uniform_filter(flat_sub**2, (20,20))
    # win_var = win_sqr_mean - win_mean**2

    # gain = (flat-1010) / win_var
    # # plt.imshow(i)
    # # plt.clim(np.percentile(i, [0,90]))
    # plt.hist(gain.flatten(), bins=np.linspace(0,2,100), density=True)
    # plt.show()
    # endd

    flats = []
    for flat_path in flats_paths:
        info = mr.info_from_fits(flat_path, telescope=station.telescope, minimal=True)
        flats.append((info['ccd_adu'] / info['ccd_adu'].mean()).astype(np.float32))

    median_flat = np.median(flats, axis=0)
    return median_flat


def calibrate_dark(darks_dir: str, median_bias: np.ndarray) -> np.ndarray:
    paths = [os.path.join(darks_dir, f) for f in os.listdir(darks_dir)]
    darks = []
    int_times = []
    for dark_path in paths:
        info = mr.info_from_fits(dark_path, telescope=station.telescope, minimal=True)
        darks.append(info['ccd_adu'])
        int_times.append(info['integration_time'])
    darks = np.array(darks)
    int_times = np.array(int_times).reshape(-1, 1, 1)

    median_dark_sub_bias = darks - median_bias
    median_dark = np.median(median_dark_sub_bias.astype(np.float32) / int_times, axis=0)

    outlier = median_dark > np.percentile(np.abs(median_dark), 95)
    dark_noise_lambda = median_dark[~outlier].mean()  # ADU / sec

    return median_dark, dark_noise_lambda


def calibrate_bias(bias_dir: str) -> tuple[np.ndarray, float]:
    paths = [os.path.join(bias_dir, f) for f in os.listdir(bias_dir)]
    imgs = []
    for p in paths:
        info = mr.info_from_fits(p, telescope=station.telescope, minimal=True)
        imgs.append(info['ccd_adu'])
    imgs = np.array(imgs)
    read_noise_std = np.std(imgs[:, 1000:1100, 1000:1100])

    return np.median(imgs, axis=0), read_noise_std


median_bias, read_noise_std = calibrate_bias(
    '/Users/liamrobinson/Library/CloudStorage/OneDrive-purdue.edu/pogs/2024_11_12/biases'
)
print(f'Read noise standard deviation (ADU/pixel) {read_noise_std:.2f}')
median_dark, dark_noise_lambda = calibrate_dark(
    '/Users/liamrobinson/Library/CloudStorage/OneDrive-purdue.edu/pogs/2024_11_12/darks',
    median_bias,
)
print(f'Dark noise mean (ADU / pixel / second) {dark_noise_lambda:.2f}')
median_flat = calibrate_flat(
    '/Users/liamrobinson/Library/CloudStorage/OneDrive-purdue.edu/pogs/2024_11_12/flats'
)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(median_bias, cmap='gray')
plt.clim(np.percentile(median_bias, [5, 95]))
plt.colorbar(cax=mrv.get_cbar_ax())
plt.title('Median Bias')
plt.xlabel('x pixels')
plt.ylabel('y pixels')
plt.subplot(1, 3, 2)
plt.imshow(median_dark, cmap='gray')
plt.colorbar(cax=mrv.get_cbar_ax())
plt.clim(np.percentile(median_dark, [5, 95]))
plt.title('Median Dark')
plt.xlabel('x pixels')
plt.ylabel('y pixels')
plt.subplot(1, 3, 3)
plt.imshow(median_flat, cmap='gray')
plt.colorbar(cax=mrv.get_cbar_ax())
plt.title('Median Flat')
plt.xlabel('x pixels')
plt.ylabel('y pixels')
plt.tight_layout()
# plt.show()

catalog = mr.GaiaSpectralStarCatalog(station)

df = pl.DataFrame()

# lights_dir = '/Users/liamrobinson/Library/CloudStorage/OneDrive-purdue.edu/pogs/2024_11_12/lights'
lights_dir = '/Users/liamrobinson/Downloads/2024_11_14/lights_100'
impaths = [os.path.join(lights_dir, f) for f in os.listdir(lights_dir)]

dfs = []
for impath in impaths:
    dfs.append(calibration_df(impath, median_flat, median_dark))
dfs = [d for d in dfs if d is not None]
df = pl.DataFrame()
for d in dfs:
    df = df.vstack(d)

in_all = df.group_by('catalog_ind').len().filter(pl.col('len') == len(dfs))
df = df.join(in_all.select('catalog_ind'), on='catalog_ind')

x = df.group_by('catalog_ind').agg(
    pl.col('predicted_flux'),
    pl.col('observed_flux'),
    pl.col('observed_flux').median().alias('observed_median'),
    pl.col('observed_flux').std(ddof=0).alias('observed_std'),
)
# observed median / observed variance
print(f"Gain ADU/e-: {(x['observed_std'] ** 2 / x['observed_median']).median()}")
print(f"Gain e-/ADU: {(x['observed_median'] / x['observed_std'] ** 2).median()}")
predicted_flux_x = np.array(x['predicted_flux'].to_list())[:, 0]
print(x)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.boxplot(
    x['observed_flux'].to_list(),
    positions=predicted_flux_x,
    widths=1000,
    showfliers=False,
)
# plt.scatter(df['predicted_flux'], df['observed_flux'], s=2, label='Observed fluxes')
# plt.scatter(predicted_flux_x, x['observed_median'], marker='+')
# plt.legend()
plt.xlabel('Predicted counts')
plt.ylabel('Observed counts')
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.subplot(1, 3, 2)
plt.scatter(x['observed_median'], x['observed_std'], s=2)
plt.xlabel(r'Median')
plt.ylabel('Observed signal standard deviation')
# plt.xscale('log')
# plt.yscale('log')
plt.grid()
plt.subplot(1, 3, 3)
plt.hist(x['observed_median'] / x['observed_std'], bins=40)  # e-/adu
plt.xlabel('Gain [e-/ADU]')
plt.ylabel('Source count')
plt.grid()
plt.tight_layout()
plt.show()

# px, py = station.telescope.j2000_unit_vectors_to_pixels(
#     look_dir_eci, up_dir_eci, uvs, add_distortion=True
# )
# plt.scatter(px, py)
