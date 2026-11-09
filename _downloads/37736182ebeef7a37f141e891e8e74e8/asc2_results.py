"""
Results for inversion paper
===========================
"""

from typing import Tuple

import numpy as np
import polars as pl
import reduction
from alive_progress import alive_bar
from pykdtree.kdtree import KDTree
from scipy.stats.qmc import Sobol

import mirage as mr


def compute_reflection_matrix(
    S: np.ndarray,
    O: np.ndarray,
    N: np.ndarray,
    A: np.ndarray,
) -> np.ndarray:
    assert S.shape[0] == O.shape[0], ValueError(
        'Size of illumination vectors must match observer vectors'
    )
    O = mr.hat(O.reshape(-1, 3))
    S = mr.hat(S.reshape(-1, 3))
    A = A.reshape(-1, 1)
    geom_count = S.shape[0]
    normal_count = N.shape[0]
    [v1_grid, v2_grid] = np.meshgrid(
        np.arange(0, geom_count), np.arange(0, normal_count)
    )
    # Grid of values for each time and facet
    ov_all = O[v1_grid.flatten(), :]  # Column vector of all observer vectors
    sv_all = S[v1_grid.flatten(), :]  # Column vector of all sun vectors
    nv_all = N[v2_grid.flatten(), :]  # Column vector of all normal vectors

    H = mr.hat(sv_all + ov_all)
    NH = mr.rdot(nv_all, H)

    cd_comp = (
        1 / np.pi * mr.rdot(nv_all, ov_all) * mr.rdot(nv_all, sv_all)
    )  # needs for LC: a * cds * cd_comp
    cs_comp = NH  # needs for LC: a * css * (ns / 8.0 + 0.25) * cs_comp ** ns

    return np.reshape(cd_comp, (geom_count, normal_count), order='F') @ A, np.reshape(
        cs_comp, (geom_count, normal_count), order='F'
    )


def find_common_row_indices(array1, array2):
    """
    Finds the indices of rows in array1 that exist as rows in array2.

    Args:
        array1 (np.ndarray): The primary array (N x M).
        array2 (np.ndarray): The lookup array (K x M).

    Returns:
        np.ndarray: A 1D array of indices (rows of array1) that are present in array2.
    """

    comparison = array1[:, None, :] == array2[None, :, :]
    all_equal = np.all(comparison, axis=2)
    rows_present_in_array2 = np.any(all_equal, axis=1)
    common_indices = np.where(rows_present_in_array2)[0]

    return common_indices


def optimize_materials(
    # true_object: mr.SpaceObject,
    row: dict,
    epsecs: np.ndarray,
    svi: np.ndarray,
    ovi: np.ndarray,
    lc_obs: np.ndarray,
    num_generations: int = 100,
    population_size: int = 200,
    tscale_lims: tuple = (0.95, 1.05),
    tdiff_lims: tuple = (-5, 5),
    cd_lims: tuple = (0.1, 1),
    cs_lims: tuple = (0.1, 1),
    n_lims: tuple = (0, 100),
    skip: int = 30,
    rtol: float = 1e-3,
):
    import jax
    from evosax.algorithms import algorithms
    from scipy.interpolate import interp1d

    eval_inds = np.arange(epsecs.size, step=skip)
    eval_inds[-1] = epsecs.size - 1  # make

    itensor = np.diag(row['itensor'])

    mat_mapping = np.array(row['mats']).squeeze()
    verts = np.array(row['v']).squeeze()
    faces = np.array(row['f']).squeeze()
    q0 = np.array(row['quat'])
    w0 = np.array(row['omega'])
    q_of_t = mr.propagate_attitude_torque_free(q0, w0, itensor, epsecs)[0]

    # deg_per_sec = np.rad2deg(np.linalg.norm(w0))

    c_of_t = mr.quat_to_dcm(q_of_t)
    svb = mr.stack_mat_mult_vec(c_of_t, svi)
    ovb = mr.stack_mat_mult_vec(c_of_t, ovi)

    lc_obs_interpolator = interp1d(
        epsecs, lc_obs, axis=0, fill_value=np.nan, bounds_error=False
    )

    obj = mr.SpaceObject(vertices_and_faces=(verts, faces))

    M = np.zeros((mat_mapping.size, mat_mapping.max() + 1), dtype=int)
    for i, m in enumerate(mat_mapping):
        M[i, m] = 1

    lc_cds = []
    cs_comps = []
    for mcol in M.T:
        minds = np.where(mcol)[0]
        cd_comp, cs_comp = compute_reflection_matrix(
            S=svb[eval_inds],
            O=ovb[eval_inds],
            N=obj.face_normals[minds],
            A=obj.face_areas[minds],
        )
        lc_cds.append(cd_comp)
        cs_comps.append(cs_comp)
    lc_cds = np.hstack(lc_cds)

    def faster_lc(cds: np.ndarray, css: np.ndarray, ns: np.ndarray):
        lc_cd = lc_cds @ cds

        lc_cs = np.zeros_like(lc_cd)

        for cs, n, mcol, cs_comp in zip(css, ns, M.T, cs_comps):
            minds = np.argwhere(mcol).flatten()
            lc_cs += (cs / np.pi * (n / 8.0 + 0.25) * cs_comp**n) @ obj.face_areas[
                minds
            ]

        return lc_cd + lc_cs

    def x_samp_to_x_mat(
        x_samp: np.ndarray, bounds: np.ndarray, k: float = 0.5
    ) -> np.ndarray:
        return (bounds[:, 1] - bounds[:, 0]) / (
            1 + np.exp(-k * np.clip(x_samp, -10, np.inf))
        ) + bounds[:, 0]

    def f(x, return_lc=False) -> float:
        if return_lc:
            _eval_inds = np.arange(epsecs.size)
        else:
            _eval_inds = eval_inds

        cds = x[:-2][0::3]
        css = x[:-2][1::3]
        ns = x[:-2][2::3]

        lc_sim = faster_lc(cds, css, ns)

        _lc = lc_obs_interpolator(x[-2] * epsecs[_eval_inds] + x[-1])
        valid = ~np.isnan(_lc)

        sim_norm = np.linalg.norm(lc_sim)
        if sim_norm == 0.0:
            return 1e10
        lc_sim *= np.linalg.norm(_lc[valid]) / sim_norm
        if return_lc:
            return lc_sim, _lc
        err = np.linalg.norm(np.abs(lc_sim[valid] - _lc[valid]))
        return err

    _x0 = [0.5, 0.5, 10]
    n_mats = mat_mapping.max() + 1
    x0 = np.concatenate(
        [*(n_mats * [_x0]), [1.00], [0.0]]
    )  # last two are time scale and time shift
    bounds = np.vstack(
        [*(n_mats * [[cd_lims, cs_lims, n_lims]]), tscale_lims, tdiff_lims]
    )

    alg = algorithms['CMA_ES']

    es = alg(population_size=population_size, solution=np.zeros_like(x0))
    params = es.default_params

    key = jax.random.key(42)
    key, subkey = jax.random.split(key)
    state = es.init(subkey, np.zeros_like(x0), params)

    best_fitness = []
    # with alive_bar(num_generations, title='Mat opt') as bar:
    for _ in range(num_generations):
        key, key_ask, key_tell = jax.random.split(key, 3)
        population, state = es.ask(key_ask, state, params)
        fitness = []

        for _x_samp in population:
            _x_mat = x_samp_to_x_mat(_x_samp, bounds)
            fitness.append(f(_x_mat))

        fitness = np.array(fitness)
        # print(fitness.min())
        best_fitness.append(fitness.min())
        if len(best_fitness) > 1:
            if abs(best_fitness[-1] - best_fitness[-2]) / best_fitness[-1] < rtol:
                break

        state, _ = es.tell(key_tell, population, fitness, state, params)
        # bar()

    # Get best solution
    sol_state = x_samp_to_x_mat(state.best_solution, bounds=bounds)
    print(
        f'{row["shape_class"]} with SDLR error: {row["sum_err"]:.2f}, material error: {state.best_fitness:.2f}'
    )
    print('hbr', np.max([np.linalg.norm(x) for x in row['v']]))

    lc_sim, _ = f(sol_state, return_lc=True)
    lc_sim /= np.sqrt(epsecs.size / eval_inds.size)

    lc_sim_big = interp1d(epsecs[eval_inds], lc_sim)(epsecs)

    return state.best_fitness, lc_sim_big, sol_state


def invert_shape_and_attitude(
    df_cal: pl.DataFrame,
    peak_finding_kwargs: dict,
    omega_range: Tuple[float],
    shape_classes: Tuple[str] = ('box_wing', 'rocket_body', 'blob'),
    n_sols_reduction: int = 100,
    n_samples_reduction: int = int(1e5),
    shgo_iters: int = 5,
    n_sols_per_shape_class: int = 100,
) -> pl.DataFrame:
    # %%
    # COMPUTING PEAKS

    df_peaks = mr.find_sdlr_peaks(
        df_cal,
        **peak_finding_kwargs,
    )

    # %%
    # Attitude and shape optimization using SHGO

    epsecs = df_cal['epsecs'].to_numpy()
    ovi = np.array(df_cal['ovi'].to_list())
    peaks = df_peaks['peak'].to_numpy()
    troughs = peaks + df_peaks['dd'].to_numpy()
    peak_inds = df_peaks['peak_ind'].to_numpy()
    trough_inds = peak_inds + df_peaks['dind'].to_numpy()
    peak_bandwidth = df_peaks['peak_bandwidth'].to_numpy()
    trough_bandwidth = df_peaks['trough_bandwidth'].to_numpy()

    node_depths = np.concatenate((peaks, troughs))
    node_widths = np.concatenate((peak_bandwidth, trough_bandwidth))
    node_inds = np.concatenate((peak_inds, trough_inds))
    node_depths = node_depths[np.argsort(node_inds).flatten()]
    node_widths = node_widths[np.argsort(node_inds).flatten()]
    node_inds = np.sort(node_inds)
    node_epsecs = epsecs[node_inds]
    node_ovi = ovi[node_inds]

    def att_shape_obj_fun(x: np.ndarray, shape_class: str):
        # bar()
        obj2, mats = mr.parametric_shape_hypercube(x, shape_class)[1]
        itensor = np.diag(obj2.principal_itensor).copy()
        sinds = itensor.argsort()
        obj2 = mr.SpaceObject(vertices_and_faces=(obj2.v[:, sinds], obj2.f))
        itensor = itensor[sinds] / itensor.min()

        best_sols = pl.DataFrame(
            reduction.best_attitude_for_obj(
                node_epsecs.tolist(),
                node_depths.tolist(),
                node_widths.tolist(),
                node_ovi.tolist(),
                tuple(itensor),
                obj2.v.tolist(),
                obj2.f.tolist(),
                n_samples_reduction,
                omega_range,
                n_sols_reduction,
            )
        )

        err = best_sols['sum_err'].mean()
        # print(best_sols['sum_err'].min(), best_sols['sum_err'].mean())
        best_sols = best_sols.with_columns(
            pl.Series(
                'x',
                best_sols.height * [x.tolist()],
            ),
            pl.Series('shape_class', len(best_sols) * [shape_class]),
            pl.Series('itensor', len(best_sols) * [list(itensor)]),
            pl.Series('v', len(best_sols) * [obj2.v.tolist()]),
            pl.Series('f', len(best_sols) * [obj2.f.tolist()]),
            pl.Series('mats', len(best_sols) * [mats.tolist()]),
        )
        sols.append(best_sols)

        # print(f'{shape_class=}, {err=}, {x=}')
        return err

    shape_sols = {}
    # att_shape_obj_fun(np.array([0, 0, 0, 0, 0, 0]), shape_class='box_wing')

    for shape_class in shape_classes:
        sols = []
        print(shape_class)
        # _ = shgo(
        #     att_shape_obj_fun,
        #     args=(shape_class,),
        #     bounds=np.array(
        #         mr.parametric_shape_state_dimension(shape_class) * [[0, 1]]
        #     ),
        #     iters=shgo_iters//2 if shape_class == 'box_wing' else shgo_iters,
        #     sampling_method='simplicial',
        #     minimizer_kwargs=dict(pgtol=1e-2),
        #     options=dict(minimize_every_iter=False),
        # )
        depth = 6 + mr.parametric_shape_state_dimension(shape_class)
        points = Sobol(d=mr.parametric_shape_state_dimension(shape_class)).random_base2(
            depth
        )
        with alive_bar(points.shape[0]) as bar:
            for point in points:
                bar()
                att_shape_obj_fun(point, shape_class)

        # print(sol)
        # print(len(sols))
        # x_mins = np.vstack((sol.x[None, :], sol.xl))
        sols = pl.concat(sols).with_row_index().sort('sum_err')
        # sols = sols.unique(['sum_err']).filter(
        #     pl.col.sum_err < pl.col.sum_err.quantile(0.1)
        # )

        # LET'S SORT OUT VERY NEARBY SOLUTIONS
        # guess_omega_dirs = mr.hat(np.array(sols['omega'].to_list()))
        # guess_omega_mags = mr.vecnorm(np.array(sols['omega'].to_list()))
        # guess_vertices = np.array(sols['v'].to_list()).reshape(sols.height, 3*len(sols['v'][0]))

        # from sklearn.cluster import DBSCAN
        # db = DBSCAN(eps=1.0, min_samples=1, p=1.0).fit(np.hstack((guess_quats, guess_omega_dirs, guess_omega_mags, guess_vertices)))
        # print(db.core_sample_indices_)
        # print(len(np.hstack((guess_quats, guess_omega_dirs, guess_omega_mags, guess_vertices))))
        # print(sols.height)
        # enddd

        dist_threshold = 0.01

        while True:
            guess_quats = mr.quat_upper_hemisphere(np.array(sols['quat'].to_list()))
            guess_omegas = np.array(sols['omega'].to_list())
            guess_itensor = np.array(sols['itensor'].to_list())[:, -2:]
            guess_hbr = np.array(
                [np.max(mr.vecnorm(v)) for v in np.array(sols['v'].to_list())]
            ).reshape(-1, 1)

            data = np.hstack((guess_quats, guess_omegas, guess_itensor, guess_hbr))
            data = (data - data.min(axis=0, keepdims=True)) / (
                data.max(axis=0, keepdims=True) - data.min(axis=0, keepdims=True)
            )

            cq = KDTree(data).query(data, k=2)[1][:, 1]
            dist2neighbor = np.max(np.abs(data - data[cq, :]), axis=1).flatten()

            sols = sols.with_columns(
                [
                    pl.Series('dist2neighbor', dist2neighbor),
                    pl.Series('neighbor_idx', cq),
                    pl.Series('self_idx', np.arange(sols.height)),
                ]
            )
            n_before = sols.height
            sols = sols.filter(
                (pl.col('dist2neighbor') >= dist_threshold)
                | (pl.col('self_idx') < pl.col('neighbor_idx'))
            )
            sols = sols.drop(['neighbor_idx', 'self_idx', 'dist2neighbor'])
            if sols.height == n_before:
                break

        sols = sols.head(n_sols_per_shape_class)
        shape_sols[shape_class] = sols
        print(sols)

        # x_sols_shape = np.array(sols['x'].to_list())

        # x_sols_shape_inds = find_common_row_indices(x_sols_shape, x_mins)
        # sols = sols.filter(pl.col.index.is_in(x_sols_shape_inds))
        # shape_sols[shape_class] = sols

    sols = pl.concat([d for d in shape_sols.values()]).sort('sum_err')

    skip_these_inds = []
    ds_est = []
    for sol in sols.iter_rows(named=True):
        q0 = np.array(sol['quat'])
        w0 = np.array(sol['omega'])
        itensor = np.array(sol['itensor'])
        q_of_t = mr.propagate_attitude_torque_free(q0, w0, np.diag(itensor), epsecs)[0]
        ovb = mr.stack_mat_mult_vec(mr.quat_to_dcm(q_of_t), ovi)
        obj2 = mr.SpaceObject(
            vertices_and_faces=(
                np.array(sol['v']).squeeze(),
                np.array(sol['f']).squeeze(),
            )
        )
        ds_est.append(mr.min_depth_for_obj(obj2, ovb).tolist())
        ds_node_est = np.array(ds_est[-1])[node_inds]
        # print(np.diff(node_depths))
        # print(np.diff(ds_node_est))
        err_test = np.mean(np.abs(np.diff(ds_node_est) - np.diff(node_depths)))
        if abs(err_test - sol['sum_err']) > 1e-2:
            print(
                'we have some errs that dont make sense', err_test, 'vs', sol['sum_err']
            )
            print(q0, w0, itensor)
            skip_these_inds.append(True)
        else:
            skip_these_inds.append(False)

    sols = (
        sols.with_columns(
            pl.Series('d_min_est', ds_est), pl.Series('skip_these', skip_these_inds)
        )
        .filter(~pl.col('skip_these'))
        .drop('skip_these')
    )
    return df_peaks, sols


# pip install --upgrade "jax[cuda12]"

# todo: implement that the object must be at least as large (in terms of HBR) as the largest range bandwidth


def main():
    # %%
    # Loading up the residuals

    "./meta/sync.sh"
    'cd ~/Documents/mirage && source .venv/bin/activate && ./meta/install-reduction.sh && python examples/16-asc-26-sim/asc2_results.py'

    np.random.seed(1)

    case_name = 'ltb'

    fname = f'scratch/f25/f25w15/data/df_{case_name}_cal.parquet'
    omega_range = (0.2, 0.3)
    df_cal = pl.read_parquet(fname)

    if case_name == 'ltb':
        peak_kwargs = dict(
            plot=True,
            peak_prominence=0.05,
            percentile_window=200,
            cut_first_last=(8, 0),
        )
    elif case_name == 'svii':
        peak_kwargs = dict(
            plot=True,
            peak_prominence=0.5,
            percentile_window=200,
            cut_first_last=(0, 0),
        )

    shgo_iters = 10  # LINEAR in the number of global samples taken! apparently.

    # true_object = mr.load_obj(df_cal['obj_path'][0])

    # p = pv.Plotter()
    # mrv.render_spaceobject(p, true_object, style='wireframe')
    # p.show()
    # endd

    df_peaks, df_sols = invert_shape_and_attitude(
        df_cal,
        peak_kwargs,
        omega_range=omega_range,
        shgo_iters=shgo_iters,
        # shape_classes=['blob'],
        # shape_classes=['box_wing'],
        # shape_classes=['rocket_body'],
    )
    df_peaks.write_parquet(
        f'examples/16-asc-26-sim/data/sdlr_peaks_{case_name}.parquet'
    )
    print(df_sols)

    # %%
    # Material optimization using CMAES

    svi = np.array(df_cal['svi'].to_list())
    ovi = np.array(df_cal['ovi'].to_list())
    lc_obs = df_cal['light_curve_ocs'].to_numpy()
    epsecs = df_cal['epsecs'].to_numpy()

    new_rows = []

    for shape_class, sols_dict in df_sols.partition_by(
        'shape_class', maintain_order=True, as_dict=True
    ).items():
        print(f'Material optimization for class {shape_class}')
        print(sols_dict)

        for row in sols_dict.rows(named=True):
            mat_error, lc_sim, opt_mat_coefs = optimize_materials(
                # true_object,
                row,
                epsecs,
                svi,
                ovi,
                lc_obs,
                tdiff_lims=(-5, 5),
                cd_lims=(0.01, 1.0),
                cs_lims=(0.01, 1.0),
                tscale_lims=(0.95, 1.05),
                num_generations=50,
                population_size=100,
            )
            new_rows.append(row.copy())
            # print(mat_error, lc_sim, opt_mat_coefs)
            new_rows[-1]['mat_error'] = mat_error
            new_rows[-1]['lc_sim'] = lc_sim.tolist()
            new_rows[-1]['opt_mat_coefs'] = opt_mat_coefs.tolist()
            # print(opt_mat_coefs.tolist())

    df_final = pl.DataFrame(new_rows)
    df_final.write_parquet(
        f'examples/16-asc-26-sim/data/sdlr_and_lc_{case_name}.parquet'
    )


if __name__ == '__main__':
    main()
