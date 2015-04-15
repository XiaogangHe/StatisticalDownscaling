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

### Data preparation
### Add adjacent (left, right, up, down) grids as features
prec_UpDown_extend = np.array([DL.extend_array_boundary(prec_UpDown[i]) for i in range(ntime)])
prec_disagg_l = prec_UpDown_extend[:, 1:-1, :-2]
prec_disagg_r = prec_UpDown_extend[:, 1:-1, 2:]
prec_disagg_u = prec_UpDown_extend[:, :-2, 1:-1]
prec_disagg_d = prec_UpDown_extend[:, 2:, 1:-1]

### Preparing the training dataset (random sample m*n grids)
prec_disagg_c_train = []
prec_disagg_l_train = []
prec_disagg_r_train = []
prec_disagg_u_train = []
prec_disagg_d_train = []
prec_fine_train = []
slope_train = []
aspect_train = []
gtopomean_train = []
gtopostd_train = []
texture_train = []
vegeType_train = []
cape180_train = []
pressfc_train = []
spfh2m_train = []
tmp2m_train = []
ugrd10m_train = []
vgrd10m_train = []

rand_row_num = 50
rand_col_num = 50

for i in range(ntime):
    ### Random choose grids
    rand_row_ind = np.random.choice(80, rand_row_num, replace=False)
    rand_col_ind = np.random.choice(72, rand_col_num, replace=False)
    ### Random sample precipitation with its adjacent grids
    grid_c = DL.get_adjacent_grids(prec_UpDown_extend[i], rand_row_ind, rand_col_ind)[0]
    grid_l = DL.get_adjacent_grids(prec_UpDown_extend[i], rand_row_ind, rand_col_ind)[1]
    grid_r = DL.get_adjacent_grids(prec_UpDown_extend[i], rand_row_ind, rand_col_ind)[2]
    grid_u = DL.get_adjacent_grids(prec_UpDown_extend[i], rand_row_ind, rand_col_ind)[3]
    grid_d = DL.get_adjacent_grids(prec_UpDown_extend[i], rand_row_ind, rand_col_ind)[4]
    prec_disagg_c_train.append(grid_c)
    prec_disagg_l_train.append(grid_l)
    prec_disagg_r_train.append(grid_r)
    prec_disagg_u_train.append(grid_u)
    prec_disagg_d_train.append(grid_d)
    prec_fine_train.append(prec_fine[i][np.ix_(rand_row_ind, rand_col_ind)])
    ### Random sample other variables 
    slope_train.append(slope[np.ix_(rand_row_ind, rand_col_ind)])
    aspect_train.append(aspect[np.ix_(rand_row_ind, rand_col_ind)])
    gtopomean_train.append(gtopomean[np.ix_(rand_row_ind, rand_col_ind)])
    gtopostd_train.append(gtopostd[np.ix_(rand_row_ind, rand_col_ind)])
    texture_train.append(texture[np.ix_(rand_row_ind, rand_col_ind)])
    vegeType_train.append(vegeType[np.ix_(rand_row_ind, rand_col_ind)])
    cape180_train.append(cape180[i][np.ix_(rand_row_ind, rand_col_ind)])
    pressfc_train.append(pressfc[i][np.ix_(rand_row_ind, rand_col_ind)])
    spfh2m_train.append(spfh2m[i][np.ix_(rand_row_ind, rand_col_ind)])
    tmp2m_train.append(tmp2m[i][np.ix_(rand_row_ind, rand_col_ind)])
    ugrd10m_train.append(ugrd10m[i][np.ix_(rand_row_ind, rand_col_ind)])
    vgrd10m_train.append(vgrd10m[i][np.ix_(rand_row_ind, rand_col_ind)])

features_train = DataFrame({
		'prec_disagg_c': np.array(prec_disagg_c_train).reshape(-1),
		'prec_disagg_l': np.array(prec_disagg_l_train).reshape(-1),
		'prec_disagg_r': np.array(prec_disagg_r_train).reshape(-1),
		'prec_disagg_u': np.array(prec_disagg_u_train).reshape(-1),
		'prec_disagg_d': np.array(prec_disagg_d_train).reshape(-1),
		'slope': np.array(slope_train).reshape(-1),
		'aspect': np.array(aspect_train).reshape(-1),
		'gtopomean': np.array(gtopomean_train).reshape(-1),
		'gtopostd': np.array(gtopostd_train).reshape(-1),
		'texture': np.array(texture_train).reshape(-1),
		'vegeType': np.array(vegeType_train).reshape(-1),
		'cape180': np.array(cape180_train).reshape(-1),
		'pressfc': np.array(pressfc_train).reshape(-1),
		'spfh2m': np.array(spfh2m_train).reshape(-1),
		'tmp2m': np.array(tmp2m_train).reshape(-1),
		'ugrd10m': np.array(ugrd10m_train).reshape(-1),
		'vgrd10m': np.array(vgrd10m_train).reshape(-1)
		})

features = DataFrame({
                'prec_disagg_c': prec_UpDown.reshape(-1),
                'prec_disagg_l': prec_disagg_l.reshape(-1),
                'prec_disagg_r': prec_disagg_r.reshape(-1),
                'prec_disagg_u': prec_disagg_u.reshape(-1),
                'prec_disagg_d': prec_disagg_d.reshape(-1),
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

prec_fine_train = DataFrame({
		'prec_fine': np.array(prec_fine_train).reshape(-1)
		})

### Need to change name
prec_fine = DataFrame({
		'prec_fine': prec_fine.reshape(-1)
		})

prec_prediction = DataFrame({
                'prec_fine': np.array([-9.99e+08]*nlat_fine*nlon_fine*ntime)
                })

### Mask out ocean : For training data
prec_validGrid_c_train = (features_train['prec_disagg_c']>-9.99e+08) & (features_train['gtopomean']>-9.99e+08)
features_train_land = features_train[prec_validGrid_c_train]
prec_fine_train_land = prec_fine_train[prec_validGrid_c_train]
prec_validGrid_l_train = features_train_land['prec_disagg_l']>-999
prec_validGrid_r_train = features_train_land['prec_disagg_r']>-999
prec_validGrid_u_train = features_train_land['prec_disagg_u']>-999
prec_validGrid_d_train = features_train_land['prec_disagg_d']>-999
features_train_land['prec_disagg_l'][~prec_validGrid_l_train] = features_train_land['prec_disagg_c'][~prec_validGrid_l_train].values
features_train_land['prec_disagg_r'][~prec_validGrid_r_train] = features_train_land['prec_disagg_c'][~prec_validGrid_r_train].values
features_train_land['prec_disagg_u'][~prec_validGrid_u_train] = features_train_land['prec_disagg_c'][~prec_validGrid_u_train].values
features_train_land['prec_disagg_d'][~prec_validGrid_d_train] = features_train_land['prec_disagg_c'][~prec_validGrid_d_train].values

### Mask out ocean : For all the data (training + test)
prec_validGrid_c = (features['prec_disagg_c']>-9.99e+08) & (features['gtopomean']>-9.99e+08)
features_land = features[prec_validGrid_c]
prec_fine_land = prec_fine[prec_validGrid_c]
prec_validGrid_l = features_land['prec_disagg_l']>-999
prec_validGrid_r = features_land['prec_disagg_r']>-999
prec_validGrid_u = features_land['prec_disagg_u']>-999
prec_validGrid_d = features_land['prec_disagg_d']>-999
features_land['prec_disagg_l'][~prec_validGrid_l] = features_land['prec_disagg_c'][~prec_validGrid_l].values
features_land['prec_disagg_r'][~prec_validGrid_r] = features_land['prec_disagg_c'][~prec_validGrid_r].values
features_land['prec_disagg_u'][~prec_validGrid_u] = features_land['prec_disagg_c'][~prec_validGrid_u].values
features_land['prec_disagg_d'][~prec_validGrid_d] = features_land['prec_disagg_c'][~prec_validGrid_d].values

### Random Forests regression
reg		= RandomForestRegressor(n_estimators=50, bootstrap=True, n_jobs=6)
time_start      = time.time()
reg.fit(features_train_land, prec_fine_train_land)
time_tr         = time.time()
prec_pre_all	= reg.predict(features_land)
time_te		= time.time()
prec_prediction['prec_fine'][prec_validGrid_c] = prec_pre_all.astype('float32')

'''
prec_fine_land = prec_fine[valid_grid]
prec_fine_land_label = prec_fine_land.values.squeeze()
prec_fine_land_label[prec_fine_land_label>0] = 1

features_land_train, features_land_test, prec_fine_land_train, prec_fine_land_test = train_test_split(features_land, prec_fine_land, test_size=0.33, random_state=42)
prec_fine_land_label_train, prec_fine_land_label_test = train_test_split(prec_fine_land_label, test_size=0.33, random_state=42)

clf		= RandomForestClassifier(n_estimators=50, bootstrap=True, n_jobs=6)
clf.fit(features_land_train, prec_fine_land_train_label)
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

