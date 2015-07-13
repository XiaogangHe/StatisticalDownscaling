#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as	plt
import dataPre_Library as DL
import datetime

### Basic information
region_info = {}
region_info['minlat'] = 25.0625
region_info['minlon'] = -88.9375
region_info['nlat_fine'] = 80 
region_info['nlon_fine'] = 72
region_info['res_fine'] = 0.125
region_info['maxlat'] = region_info['minlat'] + region_info['res_fine']*(region_info['nlat_fine']-1)
region_info['maxlon'] = region_info['minlon'] + region_info['res_fine']*(region_info['nlon_fine']-1)

date_info = {}
date_info['stime'] = datetime.datetime(2011,6,1)
date_info['ntime'] = 1000

data_info = {}
data_info['path'] = '/home/wind/hexg/Research/Data/NLDAS2/SEUS'

RF_config = {}
RF_config['rand_row_num'] = 25
RF_config['rand_col_num'] = 25
RF_config['ntree'] = 30
RF_config['njob'] = 6

features_name = {}
features_name['static'] = ['slope', 'aspect', 'gtopomean', 'gtopostd', 'texture', 'vegeType']
features_name['dynamic'] = ['prec', 'cape180', 'pressfc', 'spfh2m', 'tmp2m', 'ugrd10m', 'vgrd10m']

# Use Random Forests model for precipitation downscaling
RFDS = DL.RandomForestsDownScaling(region_info, date_info, data_info, RF_config, features_name)
prec_downscaled = RFDS.ensemble_mean()

plt.figure()
plt.imshow(np.ma.masked_equal(prec_downscaled['prec_fine'].reshape(-1, region_info['nlat_fine'], region_info['nlon_fine'])[216][::-1],-9.99e+08))
plt.show()

