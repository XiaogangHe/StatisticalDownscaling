#!/usr/bin/env python

import	numpy               as	np
import	matplotlib.pyplot   as	plt
from    dataPre_Library     import compute_variogram

nlat_fine = 80
nlon_fine = 72
ntime = 1000

data_path = '../../Data/Output/RF'
obs = np.fromfile('%s/apcpsfc_2011_JJA_SEUS.bin' % (data_path),'float32').reshape(-1, nlat_fine, nlon_fine)[:ntime]
sim = np.fromfile('%s/prec_prediction_SEUS_RF_adjacent_LargeMeteo_1deg_P_1deg_bi-linear.bin' % (data_path),'float64').reshape(-1, nlat_fine, nlon_fine)[:ntime]

prec_obs = obs[166].reshape(-1)
prec_sim = sim[166].reshape(-1)
dist_obs, gamma_obs, distFit_obs, gammaFit_obs = compute_variogram(72, 80, prec_obs)
dist_sim, gamma_sim, distFit_sim, gammaFit_sim = compute_variogram(72, 80, prec_sim)

plt.figure()
plt.plot(dist_obs, gamma_obs, 'o')
plt.plot(distFit_obs, gammaFit_obs)
plt.plot(dist_sim, gamma_sim, 'o')
plt.plot(distFit_sim, gammaFit_sim)
plt.show()

