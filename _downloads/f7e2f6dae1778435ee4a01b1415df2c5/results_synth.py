# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mirage",
#     "taichi",
# ]
#
# [tool.uv.sources]
# mirage = { path = "../../", editable = true }
# ///
"""
Synthetic Results
=================
"""

import os

import polars as pl

from mirage.papers.sdc_2025 import plot_case_results

f = os.path.join(os.environ['SRCDIR'], '..', 'examples', '15-attitude-sdc-2025', 'data')

postfix = '_no_mm'
df_in = pl.read_parquet(os.path.join(f, 'rocket_body_sdc_input.parquet'))
df_out = pl.read_parquet(
    os.path.join(f, f'rocket_body_out_final{postfix}.parquet')
).sort('fun')
df_out_prior = pl.read_parquet(
    os.path.join(f, f'rocket_body_out_prior_final{postfix}.parquet')
).sort('fun')

t_max = df_in['epsecs'][-1]

for c in ['x0', 'xk']:
    x = df_out[c].to_numpy().copy()
    x[:, 3:6] /= t_max
    df_out = df_out.with_columns(pl.Series(name=c, values=x))


for c in ['x0', 'xk']:
    x = df_out_prior[c].to_numpy().copy()
    x[:, 3:6] /= t_max
    df_out_prior = df_out_prior.with_columns(pl.Series(name=c, values=x))

# plot_case_results(
#     df_in,
#     df_out,
#     trial_name='Rocket Body - No MM',
#     conv_bound=1 / 2,
#     n=30,  # not necessary
#     cmap='viridis',
#     s_p_range=(-1, 5),
#     s_p_bins=50,
#     plot_harmonic_s_p_lines=False,
#     selection_method='likelihood',
#     color_sigma_error=False,
#     write_gifs=True,
# )


# %%
# And results for the good prior guess

plot_case_results(
    df_in,
    df_out_prior,
    trial_name='Rocket Body Prior - No MM',
    conv_bound=1 / 2,
    # n=30,
    cmap='viridis',
    s_p_range=(-1, 5),
    s_p_bins=50,
    plot_harmonic_s_p_lines=False,
    selection_method='likelihood',
    color_sigma_error=False,
    write_gifs=True,
)
