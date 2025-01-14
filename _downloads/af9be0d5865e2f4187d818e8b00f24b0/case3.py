"""
Case 3
======
"""

import mirage.papers as mrp

input_path = '/Users/liamrobinson/Documents/maintained-research/mirage/examples/13-attitude-2024/data/box_wing_input.parquet'
output_path = '/Users/liamrobinson/Documents/maintained-research/mirage/examples/13-attitude-2024/data/box-wing-known-itensor-out.parquet'

mrp.attitude_paper.run_inversion(
    input_path,
    output_path,
    n_particles=int(1e4),
    dimx=6,
    loss_function='log-likelihood',
)
