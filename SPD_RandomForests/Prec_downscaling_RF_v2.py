#!/usr/bin/env python

import	numpy				as	np
import	matplotlib.pyplot		as	plt
import	time
import	os
import	dataPre_Library			as	DL
from	pandas				import	DataFrame
from	sklearn.ensemble		import	RandomForestRegressor
from	sklearn.ensemble		import	RandomForestClassifier
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
ntime = 1000

data_path	= '/home/wind/hexg/Research/Data/NLDAS2/SEUS'

### Read datasets
prec_fine = np.fromfile('%s/apcpsfc_2011_JJA_SEUS.bin' % (data_path),'float32').reshape(-1, nlat_fine, nlon_fine)[:ntime]
prec_UpDown = np.fromfile('%s/apcpsfc_UpDown_2011_JJA_SEUS.bin' % (data_path),'float32').reshape(-1, nlat_fine, nlon_fine)[:ntime]
#ntime = prec_UpDown.shape[0]

slope = np.fromfile('%s/slope_SEUS.bin' % (data_path),'float32').reshape(nlat_fine, nlon_fine)
aspect = np.fromfile('%s/aspect_SEUS.bin' % (data_path),'float32').reshape(nlat_fine, nlon_fine)
gtopomean = np.fromfile('%s/gtopomean_SEUS.bin' % (data_path),'float32').reshape(80,72)
gtopostd = np.fromfile('%s/gtopostd_SEUS.bin' % (data_path),'float32').reshape(80,72)
texture = np.fromfile('%s/texture_SEUS.bin' % (data_path),'float32').reshape(80,72)
vegeType = np.fromfile('%s/vegeType_SEUS.bin' % (data_path),'float32').reshape(80,72)
cape180 = np.fromfile('%s/cape180-0mb_2011_JJA_SEUS.bin' % (data_path),'float32').reshape(-1,80,72)[:ntime]
pressfc = np.fromfile('%s/pressfc_2011_JJA_SEUS.bin' % (data_path),'float32').reshape(-1,80,72)[:ntime]
spfh2m = np.fromfile('%s/spfh2m_2011_JJA_SEUS.bin' % (data_path),'float32').reshape(-1,80,72)[:ntime]
tmp2m = np.fromfile('%s/tmp2m_2011_JJA_SEUS.bin' % (data_path),'float32').reshape(-1,80,72)[:ntime]
ugrd10m = np.fromfile('%s/ugrd10m_2011_JJA_SEUS.bin' % (data_path),'float32').reshape(-1,80,72)[:ntime]
vgrd10m = np.fromfile('%s/vgrd10m_2011_JJA_SEUS.bin' % (data_path),'float32').reshape(-1,80,72)[:ntime]

features = DataFrame({
		'prec_disagg': prec_UpDown.reshape(-1),
		'slope': np.array([slope.tolist()]*ntime).reshape(-1),
		'aspect': np.array([aspect.tolist()]*ntime).reshape(-1),
		'gtopomean': np.array([gtopomean.tolist()]*ntime).reshape(-1),
		'gtopostd': np.array([gtopostd.tolist()]*ntime).reshape(-1),
		'texture': np.array([texture.tolist()]*ntime).reshape(-1),
		'vegeType': np.array([vegeType.tolist()]*ntime).reshape(-1),
		'cape180': cape180.reshape(-1),
		'pressfc': pressfc.reshape(-1),
		'spfh2m': spfh2m.reshape(-1),
		'tmp2m': tmp2m.reshape(-1),
		'ugrd10m': ugrd10m.reshape(-1),
		'vgrd10m': vgrd10m.reshape(-1)
		})

### Need to change name
prec_fine = DataFrame({
		'prec_fine': prec_fine.reshape(-1)
		})

prec_prediction = DataFrame({
                'prec_fine': np.array([-9.99e+08]*nlat_fine*nlon_fine*ntime)
                })

valid_grid = (features['prec_disagg']>-9.99e+08) & (features['gtopomean']>-9.99e+08)
features_land = features[valid_grid]
prec_fine_land = prec_fine[valid_grid]
prec_fine_land_label = prec_fine_land.values.squeeze()
prec_fine_land_label[prec_fine_land_label>0] = 1

features_land_train, features_land_test, prec_fine_land_train, prec_fine_land_test = train_test_split(features_land, prec_fine_land, test_size=0.33, random_state=42)
prec_fine_land_label_train, prec_fine_land_label_test = train_test_split(prec_fine_land_label, test_size=0.33, random_state=42)

reg		= RandomForestRegressor(n_estimators=50, bootstrap=True, n_jobs=6)
#clf		= RandomForestClassifier(n_estimators=50, bootstrap=True, n_jobs=6)
time_start      = time.time()
reg.fit(features_land_train, prec_fine_land_train)
#clf.fit(features_land_train, prec_fine_land_train_label)
time_tr         = time.time()
prec_pre_all	= reg.predict(features_land)
time_te		= time.time()
prec_prediction['prec_fine'][valid_grid] = prec_pre_all.astype('float32')

'''
figure()
im=imshow(np.ma.masked_equal(prec_prediction['prec_fine'].reshape(-1, 80, 72)[0][::-1],-9.99e+08))
for a in np.ma.masked_equal(prec_prediction['prec_fine'].reshape(-1, 80, 72),-9.99e+08):
    im.set_array(a[::-1])
    draw()


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


plt.figure(figsize=(8,8));scatter(prec_fine_land['prec_fine'], a[a>0])
plt.xlim([-5,40])
plt.ylim([-5,40])
plt.xlabel('Observed hourly rainfall (kg/m2)')
plt.ylabel('Downscaled hourly rainfall (kg/m2)')

'''

