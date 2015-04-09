#!/usr/bin/env python

import	numpy				as	np
import	matplotlib.pyplot		as	plt
import	time
import	os
import	dataPre_Library			as	DL
from	pandas				import	DataFrame
from	sklearn.ensemble		import	RandomForestRegressor
from	sklearn.metrics			import	mean_squared_error
from	sklearn.cross_validation	import	train_test_split

### Basic information
res_tar         = 0.125                                 # Target resolution
res_up		= 1.0                                   # Upscaled resolution
ratio_up        = res_up/res_tar                        # Number of smaller grids within big grid box
nlat_fine	= 80
nlon_fine	= 72
nlat_coarse	= nlat_fine/ratio_up
nlon_coarse	= nlon_fine/ratio_up
ngrid_rand	= 3000					# Number of the random grids (90% of total valid grids)

data_path	= '/home/wind/hexg/Research/Data/NLDAS2/SEUS'

### Read datasets
prec_fine = np.fromfile('%s/apcpsfc_2011_JJA_SEUS.bin' % (data_path),'float32').reshape(-1, nlat_fine, nlon_fine)
prec_UpDown = np.fromfile('%s/apcpsfc_UpDown_2011_JJA_SEUS.bin' % (data_path),'float32').reshape(-1, nlat_fine, nlon_fine)

slope = np.fromfile('%s/slope_SEUS.bin' % (data_path),'float32').reshape(nlat_fine, nlon_fine)
aspect = np.fromfile('%s/aspect_SEUS.bin' % (data_path),'float32').reshape(nlat_fine, nlon_fine)
gtopomean = np.fromfile('%s/gtopomean_SEUS.bin' % (data_path),'float32').reshape(80,72)
gtopostd = np.fromfile('%s/gtopostd_SEUS.bin' % (data_path),'float32').reshape(80,72)
texture = np.fromfile('%s/texture_SEUS.bin' % (data_path),'float32').reshape(80,72)
vegeType = np.fromfile('%s/vegeType_SEUS.bin' % (data_path),'float32').reshape(80,72)
cape180 = np.fromfile('%s/cape180-0mb_2011_JJA_SEUS.bin' % (data_path),'float32').reshape(-1,80,72)
pressfc = np.fromfile('%s/pressfc_2011_JJA_SEUS.bin' % (data_path),'float32').reshape(-1,80,72)
spfh2m = np.fromfile('%s/spfh2m_2011_JJA_SEUS.bin' % (data_path),'float32').reshape(-1,80,72)
tmp2m = np.fromfile('%s/tmp2m_2011_JJA_SEUS.bin' % (data_path),'float32').reshape(-1,80,72)
ugrd10m = np.fromfile('%s/ugrd10m_2011_JJA_SEUS.bin' % (data_path),'float32').reshape(-1,80,72)
vgrd10m = np.fromfile('%s/vgrd10m_2011_JJA_SEUS.bin' % (data_path),'float32').reshape(-1,80,72)

features = DataFrame({
		'prec_disagg': prec_UpDown[0].reshape(-1),
		'slope': slope.reshape(-1),
		'aspect': aspect.reshape(-1),
		'gtopomean': gtopomean.reshape(-1),
		'gtopostd': gtopostd.reshape(-1),
		'texture': texture.reshape(-1),
		'vegeType': vegeType.reshape(-1),
		'cape180': cape180[0].reshape(-1),
		'pressfc': pressfc[0].reshape(-1),
		'spfh2m': spfh2m[0].reshape(-1),
		'tmp2m': tmp2m[0].reshape(-1),
		'ugrd10m': ugrd10m[0].reshape(-1),
		'vgrd10m': vgrd10m[0].reshape(-1)
		})

### Need to change name
prec_fine = DataFrame({
		'prec_fine': prec_fine[0].reshape(-1)
		})

prec_prediction = DataFrame({
                'prec_fine': np.ones((5760,))*(-9.99e+08)
                })

#randindex=np.random.choice(bb.index,500,replace=False)
#features_train_land.index.difference(randindex)

valid_grid = (features['prec_disagg']>-9.99e+08) & (features['gtopomean']>-9.99e+08)
features_land = features[valid_grid]
prec_fine_land = prec_fine[valid_grid]

features_land_train, features_land_test, prec_fine_land_train, prec_fine_land_test = train_test_split(features_land, prec_fine_land, test_size=0.33, random_state=42)

reg		= RandomForestRegressor(n_estimators=50, bootstrap=True, min_samples_split=2)
time_start      = time.time()
reg.fit(features_land_train, prec_fine_land_train)
time_tr         = time.time()
prec_pre_all	= reg.predict(features_land)

prec_prediction['prec_fine'][valid_grid] = prec_pre_all.astype('float32')

plt.figure()
plt.imshow(np.ma.masked_equal(prec_fine['prec_fine'].reshape(80, 72)[::-1],-9.99e+08), vmin=0, vmax=0.75)
plt.colorbar()
plt.title('Observation')
plt.savefig('../../Figures/Obs_SEUS.png', format='PNG')

plt.figure()
plt.imshow(np.ma.masked_equal(prec_prediction['prec_fine'].reshape(80, 72)[::-1],-9.99e+08), vmin=0, vmax=0.75)
plt.colorbar()
plt.title('Downscaled')
plt.savefig('../../Figures/Downscaled_SEUS.png', format='PNG')

plt.show()
'''
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
'''

'''
### Random forests
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
#plt.savefig('../../Figures/RF/RF_pre_%s%02d_%s-%s.png'%(year, mon, res_tar, res_up), format='PNG')

plt.figure()
plt.imshow(pre_fit.reshape(ngrid_Up, ngrid_Up), vmin=cbar_min, vmax=cbar_max, interpolation='nearest')
plt.colorbar()
plt.title('Fit (RF) (%s%02d)'%(year, mon), size='x-large')
#plt.savefig('../../Figures/RF/RF_fit_%s%02d_%s-%s.png'%(year, mon, res_tar, res_up), format='PNG')

plt.figure()
plt.imshow(obs_fine.reshape(nLat,nLon), vmin=cbar_min, vmax=cbar_max, interpolation='nearest')
plt.colorbar()
plt.title('Observation (RF) (%s%02d)'%(year, mon), size='x-large')
#plt.savefig('../../Figures/RF/RF_obs_%s%02d.png'%(year, mon), format='PNG')

plt.figure()
plt.imshow(obs_train.reshape(ngrid_Up, ngrid_Up), vmin=cbar_min, vmax=cbar_max, interpolation='nearest')
plt.colorbar()
plt.title('Upscale (%s%02d)'%(year, mon), size='x-large')
#plt.savefig('../../Figures/RF/RF_obsUpscale_%s%02d_%s-%s.png'%(year, mon, res_tar, res_up), format='PNG')
plt.show()
'''
