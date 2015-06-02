#!/usr/bin/env python

import	numpy			as	np
import	matplotlib.pyplot	as	plt
import	time
import	os
import	dataPre_Library		as	DL
from	pandas			import	DataFrame
from	sklearn.ensemble	import	RandomForestRegressor
from	sklearn.metrics		import	mean_squared_error

### Basic information
year		= 1992
mon		= 6
sLat            = 0
eLat            = 32
nLat            = eLat - sLat
sLon            = 0
eLon            = 32
nLon		= eLon - sLon
res_tar         = 0.125                                 # Target resolution
res_up		= 1.0                                   # Upscaled resolution
ratio_up        = res_up/res_tar                        # Number of smaller grids within big grid box
ngrid_Up	= nLat/ratio_up

dir_data	= '/home/wind/hexg/Dropbox/Study/Princeton_2014-2015_Spring/CEE509/Data/Output/grid_32'

### Read datasets
slope = np.fromfile('/home/wind/hexg/Research/Data/NLDAS2/SEUS/slope_SEUS.bin','float32').reshape(82,74)
aspect = np.fromfile('/home/wind/hexg/Research/Data/NLDAS2/SEUS/aspect_SEUS.bin','float32').reshape(82,74)
gtopomean = np.fromfile('/home/wind/hexg/Research/Data/NLDAS2/SEUS/gtopomean_SEUS.bin','float32').reshape(82,74)
gtopostd = np.fromfile('/home/wind/hexg/Research/Data/NLDAS2/SEUS/gtopostd_SEUS.bin','float32').reshape(82,74)
texture = np.fromfile('/home/wind/hexg/Research/Data/NLDAS2/SEUS/texture_SEUS.bin','float32').reshape(82,74)
vegeType = np.fromfile('/home/wind/hexg/Research/Data/NLDAS2/SEUS/vegeType_SEUS.bin','float32').reshape(82,74)
cape180 = np.fromfile('/home/wind/hexg/Research/Data/NLDAS2/SEUS/cape180-0mb_SEUS_198206.bin','float32').reshape(82,74)
pressfc = np.fromfile('/home/wind/hexg/Research/Data/NLDAS2/SEUS/pressfc_SEUS_198206.bin','float32').reshape(82,74)
spfh2m = np.fromfile('/home/wind/hexg/Research/Data/NLDAS2/SEUS/spfh2m_SEUS_198206.bin','float32').reshape(82,74)
tmp2m = np.fromfile('/home/wind/hexg/Research/Data/NLDAS2/SEUS/tmp2m_SEUS_198206.bin','float32').reshape(82,74)
ugrd10m = np.fromfile('/home/wind/hexg/Research/Data/NLDAS2/SEUS/ugrd10m_SEUS_198206.bin','float32').reshape(82,74)
vgrd10m = np.fromfile('/home/wind/hexg/Research/Data/NLDAS2/SEUS/vgrd10m_SEUS_198206.bin','float32').reshape(82,74)

slope_train = DL.upscale(slope[::-1][:nLat,:nLon], ratio_up, ratio_up, ngrid_Up)
aspect_train = DL.upscale(aspect[::-1][:nLat,:nLon], ratio_up, ratio_up, ngrid_Up)
gtopomean_train = DL.upscale(gtopomean[::-1][:nLat,:nLon], ratio_up, ratio_up, ngrid_Up)
gtopostd_train = DL.upscale(gtopostd[::-1][:nLat,:nLon], ratio_up, ratio_up, ngrid_Up)
texture_train = DL.upscale(texture[::-1][:nLat,:nLon], ratio_up, ratio_up, ngrid_Up)
vegeType_train = DL.upscale(vegeType[::-1][:nLat,:nLon], ratio_up, ratio_up, ngrid_Up)
cape180_train = DL.upscale(cape180[::-1][:nLat,:nLon], ratio_up, ratio_up, ngrid_Up)
pressfc_train = DL.upscale(pressfc[::-1][:nLat,:nLon], ratio_up, ratio_up, ngrid_Up)
spfh2m_train = DL.upscale(spfh2m[::-1][:nLat,:nLon], ratio_up, ratio_up, ngrid_Up)
tmp2m_train = DL.upscale(tmp2m[::-1][:nLat,:nLon], ratio_up, ratio_up, ngrid_Up)
ugrd10m_train = DL.upscale(ugrd10m[::-1][:nLat,:nLon], ratio_up, ratio_up, ngrid_Up)
vgrd10m_train = DL.upscale(vgrd10m[::-1][:nLat,:nLon], ratio_up, ratio_up, ngrid_Up)

obs_fine = np.loadtxt('%s/NLDAS_obs_%s%02d.txt'%(dir_data, year, mon))
obs_coarse = np.loadtxt('%s/NLDAS_obsUpDown_%s%02d.txt'%(dir_data, year, mon))

obs_train = DL.upscale(obs_fine.reshape(nLat,nLon), ratio_up, ratio_up, ngrid_Up).reshape(-1)

features_train = DataFrame({
		'slope': slope_train.reshape(-1),
		'aspect': aspect_train.reshape(-1),
		'gtopomean': gtopomean_train.reshape(-1),
		'gtopostd': gtopostd_train.reshape(-1),
		'texture': texture_train.reshape(-1),
		'vegeType': vegeType_train.reshape(-1),
		'cape180': cape180_train.reshape(-1),
		'pressfc': pressfc_train.reshape(-1),
		'spfh2m': spfh2m_train.reshape(-1),
		'tmp2m': tmp2m_train.reshape(-1),
		'ugrd10m': ugrd10m_train.reshape(-1),
		'vgrd10m': vgrd10m_train.reshape(-1)
		})
features_test = DataFrame({
		'slope': slope[::-1][:nLat,:nLon].reshape(-1),
		'aspect':aspect[::-1][:nLat,:nLon].reshape(-1),
		'gtopomean':gtopomean[::-1][:nLat,:nLon].reshape(-1),
		'gtopostd':gtopostd[::-1][:nLat,:nLon].reshape(-1),
		'texture':texture[::-1][:nLat,:nLon].reshape(-1),
		'vegeType':vegeType[::-1][:nLat,:nLon].reshape(-1),
		'cape180': cape180[::-1][:nLat,:nLon].reshape(-1),
		'pressfc': pressfc[::-1][:nLat,:nLon].reshape(-1),
		'spfh2m': spfh2m[::-1][:nLat,:nLon].reshape(-1),
		'tmp2m': tmp2m[::-1][:nLat,:nLon].reshape(-1),
		'ugrd10m': ugrd10m[::-1][:nLat,:nLon].reshape(-1),
		'vgrd10m': vgrd10m[::-1][:nLat,:nLon].reshape(-1)
		})

### Random forests
#clf_RF		= RandomForestClassifier(n_estimators=2000,random_state=0)
reg		= RandomForestRegressor(n_estimators=50, bootstrap=True, min_samples_split=2)

time_start      = time.time()
reg.fit(features_train, obs_train)
time_tr         = time.time()
pre_te		= reg.predict(features_test)
pre_fit		= reg.predict(features_train)
time_te         = time.time()

rmse_fit	= mean_squared_error(obs_train, pre_fit)**0.5
rmse_pre	= mean_squared_error(obs_fine, pre_te)**0.5
print("MSE_fit: %.4f" % rmse_fit)
print("MSE_pre: %.4f" % rmse_pre)

### Plot
cbar_min	= obs_fine.min()
cbar_max	= obs_fine.max()

plt.figure()
plt.imshow(pre_te.reshape(nLat,nLon), vmin=cbar_min, vmax=cbar_max, interpolation='nearest')
plt.colorbar()
plt.title('Prediction (RF) (%s%02d)'%(year, mon), size='x-large')
#plt.savefig('../../Figures/RF/RF_pre_%s%02d.png'%(year, mon), format='PNG')
plt.savefig('../../Figures/RF/RF_pre_%s%02d_%s-%s.png'%(year, mon, res_tar, res_up), format='PNG')

plt.figure()
plt.imshow(pre_fit.reshape(ngrid_Up, ngrid_Up), vmin=cbar_min, vmax=cbar_max, interpolation='nearest')
plt.colorbar()
plt.title('Fit (RF) (%s%02d)'%(year, mon), size='x-large')
#plt.savefig('../../Figures/RF/RF_fit_%s%02d.png'%(year, mon), format='PNG')
plt.savefig('../../Figures/RF/RF_fit_%s%02d_%s-%s.png'%(year, mon, res_tar, res_up), format='PNG')

plt.figure()
plt.imshow(obs_fine.reshape(nLat,nLon), vmin=cbar_min, vmax=cbar_max, interpolation='nearest')
plt.colorbar()
plt.title('Observation (RF) (%s%02d)'%(year, mon), size='x-large')
plt.savefig('../../Figures/RF/RF_obs_%s%02d.png'%(year, mon), format='PNG')

plt.figure()
plt.imshow(obs_train.reshape(ngrid_Up, ngrid_Up), vmin=cbar_min, vmax=cbar_max, interpolation='nearest')
plt.colorbar()
plt.title('Upscale (%s%02d)'%(year, mon), size='x-large')
plt.savefig('../../Figures/RF/RF_obsUpscale_%s%02d_%s-%s.png'%(year, mon, res_tar, res_up), format='PNG')
plt.show()
