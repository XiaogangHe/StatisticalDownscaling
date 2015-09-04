#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import dataPre_Library as DL
import datetime

### Basic information
region_info = {}
region_info['name'] = 'NWUS'
region_info['minlat'] = 32.4375
region_info['minlon'] = -124.9375
region_info['nlat_fine'] = 80 
region_info['nlon_fine'] = 120
region_info['res_fine'] = 0.125
region_info['res_coarse'] = 0.25
region_info['maxlat'] = region_info['minlat'] + region_info['res_fine']*(region_info['nlat_fine']-1)
region_info['maxlon'] = region_info['minlon'] + region_info['res_fine']*(region_info['nlon_fine']-1)

date_info = {}
date_info['stime'] = datetime.datetime(2011,6,1,0)
date_info['ftime'] = datetime.datetime(2011,8,31,23)
date_info['ntime'] = 1000

data_info = {}
data_info['prec_scale'] = 'log'    # 'linear' or 'log'
data_info['path_RF'] = '/home/wind/hexg/Research/Data/NLDAS2'
data_info['path_NLDAS2'] = '/home/raid19/forecast/NCST/nomad6.ncep.noaa.gov/pub/raid2/wd20yx/nldas/NLDASII_Forcing'
data_info['ctl_file'] = {'dynamic': 'nldasforce-a-2011.ctl',
                         'slope': 'slope15k.ctl',
                         'aspect': 'aspect15k.ctl',
                         'gtopomean': 'gtopomean15k.ctl',
                         'gtopostd': 'gtopostd15k.ctl',
                         'texture': 'NLDAS_STATSGOpredomsoil.ctl',
                         'vegeType': 'NLDAS_UMDpredomveg.ctl'
                         }

RF_config = {}
RF_config['rand_row_num'] = 25
RF_config['rand_col_num'] = 25
RF_config['ntree'] = 30
RF_config['njob'] = 6

features_name = {}
features_name['static'] = ['slope', 'aspect', 'gtopomean', 'gtopostd', 'texture', 'vegeType']
features_name['dynamic'] = ['prec', 'cape180', 'pressfc', 'spfh2m', 'tmp2m', 'ugrd10m', 'vgrd10m']

### Use Random Forests model for precipitation downscaling
RFDS = DL.RandomForestsDownScaling(region_info, date_info, data_info, RF_config, features_name)

# RFDS.subset_cov_UpDownSample('apcpsfc')
# RFDS.prepare_regional_data()

RFDS.fit_RF()
RFDS.predict_RF_mean()
#RFDS.predict_RF_all()

# Plot the spatial pattern of the observed/downscaled precipitation
obs = RFDS.prepare_prec_fine()
pre = RFDS.read_prec_downscaled(0.25)
RFDS.imshow_prec_obs(obs, itime=2, vmax=6.4)
RFDS.imshow_prec_pre(pre, itime=2, vmax=6.4, title='Downscaled (0.25 deg)')
