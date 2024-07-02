"""
IAU-76 J2000 Reduction
======================

Converting an ITRF vector to J2000 using the IAU-76 nutation theory, comparing results to Vallado's Fundamentals of Astrodynamics 4th edition
"""

import numpy as np

import mirage as mr

# %%
# Truth values presented here are copied from Vallado Example 3-15 (pg. 230)
date = mr.utc(2004, 4, 6, 7, 51, 28, 386_009)

itrf_vallado = np.array([-1033.479_383_00, 7901.295_275_40, 6380.356_595_80])
gtod_vallado = np.array([-1033.475_03_13, 7901.305_585_6, 6380.344_532_75])
tod_vallado = np.array([5094.514_780_00, 6127.366_461_2, 6380.344_532_80])
mod_vallado = np.array([5094.028_374_50, 6127.870_816_40, 6380.248_516_40])
j2000_vallado = np.array([5102.509_600_00, 6123.011_530, 6378.136_30])
gmst_vallado = 312.809_894_3
gast_vallado = 312.806_765_4

tt_vallado = 0.042_6236_319
ut1_minus_utc_vallado = -0.4399_619
delta_psi_vallado = -0.0034108
delta_eps_vallado = 0.0020316
eps_bar_vallado = 23.4387368
eps_vallado = 23.4407685

# %%
# We can now compute these values for ourselves
tt_mine = mr.date_to_julian_centuries(date)
ut1_minus_utc_mine = mr.ut1_minus_utc(date)
gmst_mine = mr.date_to_gmst(date)
gast_mine = mr.date_to_gast(date)
delta_psi_mine, delta_eps_mine, eps_bar_mine = mr.delta_psi_delta_epsilon(date)
eps_mine = eps_bar_mine + delta_eps_mine

# %%
# We can compute the error in the assorted quantities we computed
tt_error = tt_mine - tt_vallado
ut1_minus_utc_error = ut1_minus_utc_mine - ut1_minus_utc_vallado
gmst_error = (np.rad2deg(gmst_mine) - gmst_vallado) / 360 * 86400
gast_error = (np.rad2deg(gast_mine) - gast_vallado) / 360 * 86400
delta_psi_error = (
    np.rad2deg(delta_psi_mine) - delta_psi_vallado
) * mr.AstroConstants.deg_to_arcsecond
delta_eps_error = (
    np.rad2deg(delta_eps_mine) - delta_eps_vallado
) * mr.AstroConstants.deg_to_arcsecond
eps_bar_error = (
    np.rad2deg(eps_bar_mine) - eps_bar_vallado
) * mr.AstroConstants.deg_to_arcsecond
eps_error = (np.rad2deg(eps_mine) - eps_vallado) * mr.AstroConstants.deg_to_arcsecond

# %%
# And assert that these errors are sufficiently small to the Vallado values

assert np.abs(tt_error) < 1e-6, 'TT error > 1e-6 days'
assert np.abs(ut1_minus_utc_error) < 1e-3, 'UT1-UTC error > 1e-3 sec'
assert np.abs(gmst_error) < 1e-3, 'GMST error > 1e-3 sec'
assert all(np.abs(gast_error) < 1e-3), 'GAST error > 1e-3 sec'
assert all(np.abs(delta_psi_error) < 1e-3), 'DeltaPsi error > 1e-3 arcsec'
assert all(np.abs(delta_eps_error) < 1e-3), 'DeltaEpsilon error > 1e-3 arcsec'
assert np.abs(eps_bar_error) < 1e-4, 'EpsilonBar error > 1e-4 arcsec'
assert all(np.abs(eps_error) < 1e-3), 'Epsilon error > 1e-4 arcsec'


# %%
# As well as the individual frame transformations in the chain from ITRF to J2000.
# In each transformation, we start with the truth Vallado value so that the error does not accumulate from earlier transformations. This lets us see how much error each transformation introduces by itself
gtod_mine = mr.EarthFixedFrame('itrf', 'gtod').vecs_at_dates(date, itrf_vallado)
tod_mine = mr.EarthFixedFrame('gtod', 'tod').vecs_at_dates(date, gtod_vallado)
mod_mine = mr.EarthFixedFrame('tod', 'mod').vecs_at_dates(date, tod_vallado)
j2000_mine = mr.EarthFixedFrame('mod', 'j2000').vecs_at_dates(date, mod_vallado)

# %%
# Likewise, we compute the componentwise errors in each transformation in meters

gtod_error = 1e3 * (gtod_mine - gtod_vallado)
tod_error = 1e3 * (tod_mine - tod_vallado)
mod_error = 1e3 * (mod_mine - mod_vallado)
j2000_error = 1e3 * (j2000_mine - j2000_vallado)

# %%
# And assert that these must also be small -- for now we use 1 meter as the target error accumulation per transformation
assert all(np.abs(gtod_error) < 1), 'ITRF->GTOD error > 1 m'
assert all(np.abs(tod_error) < 1), 'GTID->TOD error > 1 m'
assert all(np.abs(mod_error) < 1), 'TOD->MOD error > 1 m'
assert all(np.abs(j2000_error) < 1), 'MOD->J2000 error > 1 m'

# %%
# Let's collect these error results

print(f'{tt_error=} [days]')
print(f'{ut1_minus_utc_error=} [sec]')
print(f'{gmst_error=} [sec]')
print(f'{gast_error=} [sec]')
print(f'{delta_psi_error=} [arcsec]')
print(f'{delta_eps_error=} [arcsec]')
print(f'{eps_bar_error=} [arcsec]')
print(f'{eps_error=} [arcsec]')

print('\n')
print(f'{gtod_error=} [m]')
print(f'{tod_error=} [m]')
print(f'{mod_error=} [m]')
print(f'{j2000_error=} [m]')


# %%
# We can now perform a simultaneous transformation that deals with all the sub-rotations behind the scenes

j2000_mine_combined = mr.EarthFixedFrame('itrf', 'j2000').vecs_at_dates(
    date, itrf_vallado
)

print(f'MINE: {j2000_mine_combined}')
print(f'VALL: {j2000_vallado}')
print(
    f'Transformation error: {np.linalg.norm(j2000_mine_combined - j2000_vallado) * 1e3} [m]'
)

# %%
# Finally, let's run the transformation in reverse to make sure we can both directions

itrf_mine_combined = mr.EarthFixedFrame('j2000', 'itrf').vecs_at_dates(
    date, j2000_vallado
)

print(f'MINE: {itrf_mine_combined}')
print(f'VALL: {itrf_vallado}')
print(
    f'Transformation error: {np.linalg.norm(itrf_mine_combined - itrf_vallado) * 1e3} [m]'
)
