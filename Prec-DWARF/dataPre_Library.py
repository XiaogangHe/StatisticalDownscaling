#-*-coding:utf-8-*-

#!/usr/bin/env python

from pandas import DataFrame
from matplotlib import colors
from mpl_toolkits.basemap import Basemap
from sklearn.ensemble import RandomForestRegressor
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pandas.rpy.common as com
import pathos.multiprocessing as mp
import matplotlib.cbook as cbook
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
from rpy2.robjects import globalenv
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import grads
grads_exe = '/home/wind/hexg/opengrads/grads'
ga = grads.GrADS(Bin=grads_exe, Window=False, Echo=False)
import sys
import gc
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.patches as patches

colors_1RF = ['#f03b20', '#feb24c', '#c51b8a']
colors_2RF = ['#31a354', '#addd8e', '#67a9cf'] 
width = [0.88, 0.55, 0.33]

class RandomForestsDownScaling(object):
    
    """
    This is a class to use Random Forests for precipitation downscaling
    
    Args:
        :region_info (dic): domain information
        :date_info (dic): date information for simulation 
        :data_info (dic): path etc
        :features_name (list): list of feature names (static + dynamic)
    
    """

    def __init__(self, region_info, date_info, data_info, RF_config, features_name):
        self._minlat_CONUS = 25.0625
        self._minlon_CONUS = -124.9375
        self._nlat_fine_CONUS = 224
        self._nlon_fine_CONUS = 464
        self._region_name = region_info['name']
        self._minlat = region_info['minlat']
        self._minlon = region_info['minlon']
        self._maxlat = region_info['maxlat']
        self._maxlon = region_info['maxlon']
        self._nlat_fine = region_info['nlat_fine']
        self._nlon_fine = region_info['nlon_fine']
        self._res_fine = region_info['res_fine']
        self._res_coarse = region_info['res_coarse']
        self._scaling_ratio = self._res_coarse/self._res_fine
        self._stime = date_info['stime']
        self._ftime = date_info['ftime']
        self._ntime = date_info['ntime']
        self._prec_scale = data_info['prec_scale']
        self._path_RF = data_info['path_RF']
        self._path_RF_subregion = data_info['path_RF'] + '/' + region_info['name']
        self._path_NLDAS2 = data_info['path_NLDAS2']
        self._ctl_file = data_info['ctl_file']
        self._seperateRF = RF_config['seperateRF']
        self._rand_row_num = RF_config['rand_row_num']
        self._rand_col_num = RF_config['rand_col_num']
        self._ntree = RF_config['ntree']
        self._njob = RF_config['njob']
        self._features_static = features_name['static']
        self._features_dynamic = features_name['dynamic']

        self.reg = RandomForestRegressor(n_estimators=self._ntree, bootstrap=True, oob_score=True, n_jobs=self._njob)
        self.reg_ext = RandomForestRegressor(n_estimators=self._ntree, bootstrap=True, oob_score=True, n_jobs=self._njob)

    def subset_prec(self):
        """
        Extract the fine-resolution sub-region precipitation from CONUS using PyGrads

        """

        # Open access to the file
        ga("open %s/%s" % (self._path_NLDAS2, self._ctl_file)) 

        # Output the data 
        ga("set lat %s %s" % (self._minlat, self._maxlat)) 
        ga("set lon %s %s" % (self._minlon, self._maxlon)) 
        ga("set time %s %s" % (self._stime.strftime("%Hz%d%b"), self._ftime.strftime("%Hz%d%b")))
        ga("set gxout fwrite")
        ga("set fwrite prec_2011_JJA_%s.bin" % (self._region_name))
        ga("d apcpsfc")
        ga("disable fwrite")

        # Close access to all files
        ga("close 1")

    def subset_cov_static(self):
        """
        Extract the fine-resolution static covariates from CONUS using PyGrads

        """

        for static_feature in self._features_static:
            # Open access to the file
            ga("open %s/%s" % (self._path_RF, self._ctl_file[static_feature])) 

            # Output the data 
            ga("set lat %s %s" % (self._minlat+0.0005, self._maxlat-0.0005)) 
            ga("set lon %s %s" % (self._minlon+0.0005, self._maxlon-0.0005)) 
            ga("set gxout fwrite")
            ga("set fwrite %s_%s.bin" % (static_feature, self._region_name))
            ga("d %s" % (static_feature))
            ga("disable fwrite")

            # Close access to all files
            ga("close 1")

        return

    def subset_cov_UpDownSample(self, var):
        """
        Extract the sub-region synthetic covariates from CONUS using PyGrads through the upsampling and downsampling

        Args:
            :var (str): variable name in the GrADS control file 

        """

        # Open access to the file
        nlat_coarse = self._nlat_fine/self._scaling_ratio + 2
        nlon_coarse = self._nlon_fine/self._scaling_ratio + 1

        ga("open %s/%s" % (self._path_NLDAS2, self._ctl_file['dynamic'])) 

        # Set to new region
        ga("set lat %s %s" % (self._minlat, self._maxlat+self._res_fine)) 
        ga("set lon %s %s" % (self._minlon, self._maxlon+self._res_fine)) 

        ga("define mask=const(%s, 1)" % (var))
        ga("set time %s %s" % (self._stime.strftime("%Hz%d%b"), self._ftime.strftime("%Hz%d%b")))

        # Upsample and downsample 
        ga("define up=re(%s, %s, linear, %s, %s, %s, linear, %s, %s, ba)" \
                % (var, nlon_coarse, self._minlon, self._res_coarse, nlat_coarse, self._minlat, self._res_coarse)) 
        ga("define down=re(up, %s, linear, %s, %s, %s, linear, %s, %s, ba)" \
                % (self._nlon_fine+1, self._minlon, self._res_fine, self._nlat_fine+1, self._minlat, self._res_fine))
        ga("define up=re(down, %s, linear, %s, %s, %s, linear, %s, %s, ba)" \
                % (nlon_coarse, self._minlon, self._res_coarse, nlat_coarse, self._minlat, self._res_coarse)) 
        ga("define down=re(up, %s, linear, %s, %s, %s, linear, %s, %s, bl)" \
                % (self._nlon_fine+1, self._minlon, self._res_fine, self._nlat_fine+1, self._minlat, self._res_fine))
        ga("define downMask=maskout(down, mask-0.1)")

        # Output the data 
        ga("set lat %s %s" % (self._minlat, self._maxlat)) 
        ga("set lon %s %s" % (self._minlon, self._maxlon)) 
        ga("set gxout fwrite")
        ga("set fwrite %s_UpDown_%sdeg_2011_JJA_%s_bi-linear.bin" % (var, self._res_coarse, self._region_name))
        ga("d downMask")
        ga("disable fwrite")

        # Close access to all files
        ga("close 1")

        return

    def extend_array_boundary(self, inArr):
        """
        Add the boundary to the original array 

        Change shape from (x,y) to (x+2,y+2)
        """

        add_row = np.r_[[inArr[0]], inArr, [inArr[-1]]]
        add_row_col = np.c_[add_row[:,0], add_row, add_row[:,-1]]
        return add_row_col

    def get_adjacent_grids(self, extendArr, rand_row, rand_col):
        """
        Get the adjacent grids
    
        Input: Extended array (adding boundarys to the original array)
    
        """
        grid_central = extendArr[1:-1, 1:-1][np.ix_(rand_row, rand_col)]
        grid_left = extendArr[1:-1][np.ix_(rand_row, rand_col)]
        grid_right = extendArr[1:-1][np.ix_(rand_row, rand_col+2)]
        grid_up = extendArr[:, 1:-1][np.ix_(rand_row, rand_col)]
        grid_down = extendArr[:, 1:-1][np.ix_(rand_row+2, rand_col)]
        return grid_central, grid_left, grid_right, grid_up, grid_down

    def get_DOYs(self):
        """
        Create the covariance (day of year (DOY)) for each time step
    
        """
        dates = pd.date_range(self._stime, periods=self._ntime, freq='H') 
        DOY = dates.dayofyear 
        DOYs = np.array([np.ones((self._nlat_fine, self._nlon_fine))*DOY[i] for i in xrange(self._ntime)])
        return DOYs

    def get_lons_lats(self):
        """
        Create the covariance (latitude and longitude) for each time step
    
        """
        lons = np.arange(self._minlon, self._minlon+self._res_fine*self._nlon_fine, self._res_fine)
        lats = np.arange(self._minlat, self._minlat+self._res_fine*self._nlat_fine, self._res_fine)
        lons, lats = np.meshgrid(lons, lats)
        lons = np.array([lons]*self._ntime)
        lats = np.array([lats]*self._ntime)
        return lons, lats

    def get_closest_distance(self, prec_2d):
        """
        Get the closest distance to the surrounding dry grid cells
        Input:  prec_2d: 2-d array
    
        """
        from sklearn.neighbors import KDTree
        
        # Prepare the coordinates
        lons, lats = self.get_lons_lats()
        lat_lon = np.hstack([lats[0].reshape(-1,1), lons[0].reshape(-1,1)])

        # Find out the dry/wet/ocean grids
        loc_ocean = prec_2d == -9.99e+08
        loc_dry = prec_2d == 0
        loc_wet = prec_2d > 0
        loc_ocean = loc_ocean.reshape(1,-1).squeeze()
        loc_dry = loc_dry.reshape(1,-1).squeeze()
        loc_wet = loc_wet.reshape(1,-1).squeeze()

        # Use KDTree to find the closest distance
        if len(lat_lon[loc_wet]) == 0:
            dist_wet = 0
        elif len(lat_lon[loc_dry]) == 0:
            dist_wet = 10    # Arbitrary value
        else:
            tree = KDTree(lat_lon[loc_dry], leaf_size=2)
            dist_wet = tree.query(lat_lon[loc_wet], k=1)[0]

        dist_all = np.array([-9.99e+08]*self._nlat_fine*self._nlon_fine)
        dist_all[loc_dry] = 0
        dist_all[loc_wet] = dist_wet
        dist_all = dist_all.reshape(self._nlat_fine, self._nlon_fine)
        
        return dist_all

    def get_closest_distance_CONUS(self, resolution=None, threshold=0):
        """
        Calculate the closest distance to the surrounding dry grid cells for the CONUS

        Args:
            :resolution (str): coarse resolution 
    
        """

        resolution = resolution or self._res_coarse
        prec_UpDown_CONUS = self.read_prec_UpDown_CONUS(resolution)

        # Precipitatin will be set to 0 if under threshold
        if threshold ==0:
            pass
        else:
            prec_UpDown_CONUS[(0<prec_UpDown_CONUS) & (prec_UpDown_CONUS<threshold)]=0

        dist_CONUS = np.array([self.get_closest_distance(prec_UpDown_CONUS[i]) for i in xrange(self._ntime)]) 
        dist_CONUS.tofile('%s/distance_syn_%sdeg_2011_JJA_CONUS_threshold.bin' % (self._path_RF, resolution))

        return dist_CONUS

    def read_prec_UpDown_region(self, resolution=None):
        """
        This function is used to read the synthetic regional precipitation
    
        Args:
            :resolution (str): coarse resolution 
    
        """

        resolution = resolution or self._res_coarse
        prec_UpDown_region = np.fromfile('%s/prec_UpDown_%sdeg_2011_JJA_%s_bi-linear.bin' % \
                (self._path_RF_subregion, resolution, self._region_name),'float32').reshape(-1, self._nlat_fine, self._nlon_fine)[:self._ntime]

        return prec_UpDown_region

    def read_prec_UpDown_CONUS(self, resolution=None):
        """
        This function is used to read the synthetic CONUS precipitation
    
        Args:
            :resolution (str): coarse resolution 
    
        """

        resolution = resolution or self._res_coarse
        prec_UpDown_CONUS = np.fromfile('%s/apcpsfc_UpDown_%sdeg_2011_JJA_CONUS_bi-linear.bin' % \
                (self._path_RF, resolution),'float32').reshape(-1, self._nlat_fine_CONUS, self._nlon_fine_CONUS)[:self._ntime]

        return prec_UpDown_CONUS

    def subset_closest_distance_CONUS(self, resolution=None):
        """
        Subset the closest distance from CONUS 

        Args:
            :resolution (str): coarse resolution 
    
        """
        
        resolution = resolution or self._res_coarse
        sInd_lat = (self._minlat-self._minlat_CONUS)/self._res_fine
        sInd_lon = (self._minlon-self._minlon_CONUS)/self._res_fine
        dist_CONUS = np.fromfile('%s/distance_syn_%sdeg_2011_JJA_CONUS.bin' % (self._path_RF, resolution)).reshape(-1, self._nlat_fine_CONUS, self._nlon_fine_CONUS)
        dist = dist_CONUS[:, sInd_lat:sInd_lat+self._nlat_fine, sInd_lon:sInd_lon+self._nlon_fine]

        return dist

    def prepare_regional_data(self):
        """
        Subset regional data and save to the local disk
    
        """
        # self.subset_prec()
        # self.subset_cov_static()
        
        # !!! Need to add process other covariates

        if os.path.exists(self._path_RF_subregion) == False:
            os.mkdir(self._path_RF_subregion)

        os.system("mv *.bin %s" % (self._path_RF_subregion))

        return 

    def prepare_covariates(self):
        """
        Prepare precipitation observations
    
        """

        # Add static covariates
        self.features_dic = {static_feature: \
                np.fromfile('%s/%s_%s.bin' % (self._path_RF_subregion, static_feature, self._region_name),'float32').reshape(self._nlat_fine, self._nlon_fine) \
                for static_feature in self._features_static} 

        # Add dynamic covariates
        features_dic_dynamic = {dynamic_feature: \
                np.fromfile('%s/%s_UpDown_%sdeg_2011_JJA_%s_bi-linear.bin' % \
                (self._path_RF_subregion, dynamic_feature, self._res_coarse, self._region_name),'float32').reshape(-1, self._nlat_fine, self._nlon_fine)[:self._ntime] \
                for dynamic_feature in self._features_dynamic} 
        self.features_dic.update(features_dic_dynamic)

        # Log transformation
        if self._prec_scale == 'log':
            self.features_dic['prec'][self.features_dic['prec']>0] = np.log10(self.features_dic['prec'][self.features_dic['prec']>0]) 

        prec_UpDown = self.features_dic['prec']
        self.prec_UpDown_extend = np.array([self.extend_array_boundary(prec_UpDown[i]) for i in xrange(self._ntime)])
        self.features_dic['prec_disagg_l'] = self.prec_UpDown_extend[:, 1:-1, :-2]
        self.features_dic['prec_disagg_r'] = self.prec_UpDown_extend[:, 1:-1, 2:]
        self.features_dic['prec_disagg_u'] = self.prec_UpDown_extend[:, :-2, 1:-1]
        self.features_dic['prec_disagg_d'] = self.prec_UpDown_extend[:, 2:, 1:-1]

        # Add DOY (day of year) as covariates
        self.features_dic['DOY'] = self.get_DOYs()

        # Add lats & lons as covariates
        self.features_dic['lons'] = self.get_lons_lats()[0]
        self.features_dic['lats'] = self.get_lons_lats()[1]

        # Add the closest distance to dry grid cells as covariates
        # self.features_dic['distance'] = np.array([self.get_closest_distance(prec_UpDown[i]) for i in xrange(self._ntime)]) 
        self.features_dic['distance'] = self.subset_closest_distance_CONUS(self._res_coarse)

        # return self.features_dic
        return 

    def prepare_prec_fine(self, prec_scale=None):
        """
        Prepare precipitation observations at fine resolution
    
        """
        prec_fine = np.fromfile('%s/prec_2011_JJA_%s.bin' 
                                     % (self._path_RF_subregion, self._region_name), 'float32'). \
                                     reshape(-1, self._nlat_fine, self._nlon_fine)[:self._ntime]

        prec_scale = prec_scale or self._prec_scale
        if prec_scale == 'linear':
            return prec_fine
        if prec_scale == 'log':
            prec_fine[prec_fine>0] = np.log10(prec_fine[prec_fine>0])
            return prec_fine

    def mask_out_ocean(self, covariates_df, response_df):
        """
        This function can be used to mask out ocean grid cells
    
        Args:
            :covariates_df (df): Dataframe for features
            :response_df (df): Dataframe for fine resolution precipitation
    
        """

        validGrid_c = (covariates_df['prec_disagg_c']>-9.99e+08) & (covariates_df['gtopomean']>-9.99e+08) & (covariates_df['cape180']>-9.99e+08)
        covariates_land_df = covariates_df[validGrid_c]
        response_land_df = response_df[validGrid_c]

        for prec_feature_adjacent in self.prec_feature:
            if prec_feature_adjacent == 'prec_disagg_c':
                pass
            else:
                prec_validGrid_lrud = covariates_land_df[prec_feature_adjacent]>-999
                covariates_land_df.loc[~prec_validGrid_lrud, (prec_feature_adjacent)] = covariates_land_df.loc[~prec_validGrid_lrud, ('prec_disagg_c')].values

        return covariates_land_df, response_land_df 

    def prepare_training_data(self):
        """
        Prepare training datasets
    
        """

        prec_fine = self.prepare_prec_fine(self._prec_scale)
        self.prepare_covariates()

        features_name_train = self.features_dic.keys()
        features_name_train.remove('prec')
        features_train_dic = {feature_train: [] for feature_train in features_name_train} 
        features_train_dic['prec_disagg_c'] = [] 
        self.prec_fine_train = [] 
        
        dynamic_feature_update = [name for name in self._features_dynamic if name != 'prec']
        dynamic_feature_update.extend(['DOY', 'lons', 'lats', 'distance'])            
        self.prec_feature = ['prec_disagg_c', 'prec_disagg_l', 'prec_disagg_r', 'prec_disagg_u', 'prec_disagg_d']

        for i in xrange(self._ntime):
            ### Random choose grids
            # rand_row_ind = np.random.choice(self._nlat_fine, self._rand_row_num, replace=False)
            # rand_col_ind = np.random.choice(self._nlon_fine, self._rand_col_num, replace=False)
            rand_row_ind = np.random.choice(self._nlat_fine, self._rand_row_num, replace=True)
            rand_col_ind = np.random.choice(self._nlon_fine, self._rand_col_num, replace=True)

            ### Random sample fine precipitation
            self.prec_fine_train.append(prec_fine[i][np.ix_(rand_row_ind, rand_col_ind)])

            ### Random sample coarse precipitation with its adjacent grids
            for prec_ind, prec_name in enumerate(self.prec_feature):
                grid_loc = self.get_adjacent_grids(self.prec_UpDown_extend[i], rand_row_ind, rand_col_ind)[prec_ind]
                features_train_dic[prec_name].append(grid_loc)

            ### Random sample other variables 
            for static_feature in self._features_static:
                features_train_dic[static_feature].append(self.features_dic[static_feature][np.ix_(rand_row_ind, rand_col_ind)])
            
            for dynamic_feature in dynamic_feature_update:
                features_train_dic[dynamic_feature].append(self.features_dic[dynamic_feature][i][np.ix_(rand_row_ind, rand_col_ind)])

        ### Create dataframe for features
        self.features_name_all = features_train_dic.keys()
        features_train_df = DataFrame({feature_all: np.array(features_train_dic[feature_all]).reshape(-1) for feature_all in self.features_name_all}) 
        features_df = DataFrame({static_feature: np.array([self.features_dic[static_feature].tolist()]*self._ntime).reshape(-1) for static_feature in self._features_static}) 
        for dynamic_feature in dynamic_feature_update + self.prec_feature:
            if dynamic_feature == 'prec_disagg_c':
                features_df[dynamic_feature] = self.features_dic['prec'].reshape(-1)
            else:
                features_df[dynamic_feature] = self.features_dic[dynamic_feature].reshape(-1)

        ### Create dataframe for precipitation
        prec_fine_train_df = DataFrame({'prec_fine': np.array(self.prec_fine_train).reshape(-1)})
        prec_fine_df = DataFrame({'prec_fine': prec_fine.reshape(-1)})

        ### Mask out ocean grid cells
        self.features_train_land_df, self.prec_fine_train_land_df = self.mask_out_ocean(features_train_df, prec_fine_train_df)
        # self.features_land_df = self.mask_out_ocean(features_df, prec_fine_df)[0]
        self.features_land_df, self.prec_fine_land_df = self.mask_out_ocean(features_df, prec_fine_df)
        self.features_land_df = self.features_land_df.sort_index(axis=1)

        # return self.features_train_land_df, self.prec_fine_train_land_df, self.features_land_df, self.prec_fine_land_df 
        return 

    def fit_RF(self):
        """
        Fit random forests using the training data

        Args:
            :seperate (bool): False: fit one RF; True: fit seperated RF accounting for extreme values
    
        """

        self.prepare_training_data()
        self.reg.fit(self.features_train_land_df, np.ravel(self.prec_fine_train_land_df))

        ### Fit for extreme precipitation
        if self._seperateRF == True:
            ### Set the threshold to be the 99.9% percentile
            threshold_ext = np.percentile(self.prec_fine_land_df['prec_fine'], 99.95)
            ### Subset the extreme values
            prec_ext_ind = self.prec_fine_land_df['prec_fine'] > threshold_ext
            self.prec_ext_ind = self.prec_fine_land_df[prec_ext_ind].index
            self.prec_ext_ind_tr = np.random.choice(self.prec_ext_ind, round(0.3*len(self.prec_ext_ind)), replace=False)
            self.reg_ext.fit(self.features_land_df.loc[self.prec_ext_ind_tr], np.ravel(self.prec_fine_land_df.loc[self.prec_ext_ind_tr]))

        del self.features_train_land_df
        del self.prec_fine_train_land_df
        gc.collect()

        return

    def predict_RF_mean(self):
        """
        Use random forests to train and downscale coarse resolution precipitation
    
        """
        
        prec_pred_df = DataFrame({'prec_fine': np.array([-9.99e+08]*self._nlat_fine*self._nlon_fine*self._ntime)})
        prec_pre_all = self.reg.predict(self.features_land_df)
        prec_pred_df['prec_fine'][self.features_land_df.index] = prec_pre_all.astype('float32')

        ### Predict for extreme precipitation
        if self._seperateRF == True:
            prec_pre_ext = self.reg_ext.predict(self.features_land_df.loc[self.prec_ext_ind])
            prec_pred_df.loc[self.prec_ext_ind] = prec_pre_ext.astype('float32').reshape(-1, 1)

        if self._prec_scale == 'log':
            ind = prec_pred_df['prec_fine'] != 0
            prec_pred_df['prec_fine'][ind] = np.power(10, prec_pred_df['prec_fine'][ind])
 
        prec_pred_df['prec_fine'].values.tofile('%s/prec_prediction_%s_RF_adjacent_LargeMeteo_%sdeg_P_%sdeg_2RF.bin' 
                                                      % (self._path_RF_subregion, self._region_name, self._res_coarse, self._res_coarse))
        return prec_pred_df

    def predict_RF_all(self):
        """
        Prediction from each individual decision tree
    
        """

        prec_pred_land_trees = np.array([self.reg.estimators_[i].predict(self.features_land_df.values) for i in xrange(self._ntree)])

        row_index = range(self._nlat_fine*self._nlon_fine*self._ntime)
        tree_index = ['tree_%02d'%(i) for i in xrange(self._ntree)]
        prec_pred_all_trees = DataFrame(index=row_index, columns=tree_index)
        prec_pred_all_trees = prec_pred_all_trees.fillna(-9.99e+08)

        for i in xrange(self._ntree):
            print i
            prec_pred_all_trees['tree_%02d'%(i)][self.features_land_df.index] = prec_pred_land_trees[i].astype('float32')

        tree_hour_image = np.array([prec_pred_all_trees['tree_%02d'%(i)].values.reshape(-1, self._nlat_fine, self._nlon_fine) for i in xrange(self._ntree)])
        tree_hour_image.tofile('%s/tree_hour_image_%sdeg_%s.bin' % (self._path_RF_subregion, self._res_coarse, self._region_name))

        return

    def read_prec_downscaled(self, resolution=None):
        """
        This function is used to read the downscaled precipitation from output file
    
        Args:
            :resolution (str): coarse resolution 
    
        """

        resolution = resolution or self._res_coarse
        prec_downscaled = np.fromfile('%s/prec_prediction_%s_RF_adjacent_LargeMeteo_%sdeg_P_%sdeg_2RF.bin' % 
                          (self._path_RF_subregion, self._region_name, resolution, resolution),'float64').reshape(-1, self._nlat_fine, self._nlon_fine)

        return prec_downscaled

    def compute_variogram(self, data, itime, psill=0, vrange=40, nugget=0):
        """
        Compute the semi-variogram using R's 'gstat' package

        Input:  data: 3-d array (time, lat, lon)
                itime (int): ith time step
        """

        ##### Load R packages
        r('library("gstat")')
        r('library("sp")')

        lon = r("lon <- expand.grid(1:%d, 1:%d)[,1]" % (self._nlon_fine, self._nlat_fine))
        lat = r("lat <- expand.grid(1:%d, 1:%d)[,2]" % (self._nlon_fine, self._nlat_fine))
        lon = np.array(lon)
        lat = np.array(lat)
        
        data = data[itime].reshape(-1)
        ind = data != -9.99e+08
        data = data[ind]
        lon = lon[ind]
        lat = lat[ind]

        ##### Convert numpy to R format
        r.assign("data", data)
        r.assign("lon", lon)
        r.assign("lat", lat)

        ##### Fit variogram
        r("d = data.frame(lon=lon, lat=lat, prec=data)")
        r("coordinates(d)<-~lon+lat")
        r('vg <- variogram(prec~1, d)')
        r("vg.fit <- fit.variogram(vg, vgm(%s, 'Exp', %s, %s))" % (psill, vrange, nugget))

        dist = np.array(r("vg$dist"))
        gamma = np.array(r("vg$gamma"))
        dist_fit = np.array((r('variogramLine(vg.fit, %s)$dist' % (vrange))))
        gamma_fit = np.array((r('variogramLine(vg.fit, %s)$gamma' % (vrange))))

        return dist, gamma, dist_fit, gamma_fit
        return {'dist':dist, 'gamma':gamma, 'dist_fit':dist_fit, 'gamma_fit':gamma_fit}

    def score_RMSE_R2(self, resolution=None):
        """
        This function is used to calculate the RMSE value
    
        Args:
            :resolution (str): coarse resolution 
    
        """

        prec_observed = self.prepare_prec_fine()
        prec_downscaled = self.read_prec_downscaled(resolution)

        prec_observed_valid = prec_observed[prec_downscaled > -9.99e+08]
        prec_downscaled_valid = prec_downscaled[prec_downscaled > -9.99e+08]

        score_RMSE = mean_squared_error(prec_observed_valid, prec_downscaled_valid)**0.5
        score_R2 = r2_score(prec_observed_valid, prec_downscaled_valid)

        return score_RMSE, score_R2

    def score_QQ(self, resolution=None, prec_scale=None):
        """
        Use this function to output the sample quantiles for observed and downscaled precipitation
    
        Args:
            :resolution (str): coarse resolution 
    
        """

        import statsmodels.api as sm
        resolution = resolution or self._res_coarse
        prec_scale = prec_scale or self._prec_scale

        prec_observed = self.prepare_prec_fine(prec_scale)
        prec_downscaled = self.read_prec_downscaled(resolution)

        prec_observed_valid = prec_observed[(prec_downscaled > -9.99e+08) & (prec_observed > -9.99e+08)]
        prec_downscaled_valid = prec_downscaled[(prec_downscaled > -9.99e+08) & (prec_observed > -9.99e+08)]

        pp_observed = sm.ProbPlot(prec_observed_valid)
        pp_downscaled = sm.ProbPlot(prec_downscaled_valid)

        plt.figure()
        plt.scatter(pp_downscaled.sample_quantiles, pp_observed.sample_quantiles)
        # fig = pp_observed.qqplot(line='45', other=pp_downscaled)
        plt.xlabel('Downscaled')
        plt.ylabel('Obs')
        plt.title('%s deg' % (resolution))
        plt.show()

        pp_observed.sample_quantiles.tofile('%s/quantiles_obsmask_LargeMeteo_%sdeg_P_%sdeg_%s_2RF.bin' % (self._path_RF_subregion, resolution, resolution, self._region_name))
        pp_downscaled.sample_quantiles.tofile('%s/quantiles_downscaled_LargeMeteo_%sdeg_P_%sdeg_%s_2RF.bin' % (self._path_RF_subregion, resolution, resolution, self._region_name))

        return 

    def score_ROC(self, resolution=None):
        """
        Use this function to output the sample quantiles for observed and downscaled precipitation
    
        Args:
            :resolution (str): coarse resolution 
    
        """

        from sklearn.metrics import roc_curve, auc, accuracy_score
        resolution = resolution or self._res_coarse

        prec_observed = self.prepare_prec_fine()
        prec_downscaled = self.read_prec_downscaled(resolution)

        prec_observed_valid = prec_observed[prec_downscaled > -9.99e+08]
        prec_downscaled_valid = prec_downscaled[prec_downscaled > -9.99e+08]

        ### Calculate and save ROC metrics
        fpr, tpr, thresholds = roc_curve(prec_observed_valid, prec_downscaled_valid, pos_label=2)
        roc_auc = auc(fpr,tpr)

        plt.figure()
        plt.plot(fpr, tpr, linewidth=2.5, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim([0,0.2])
        plt.ylim([0.4,1.05])
        plt.legend(loc='best')
        plt.show()

        np.savez('%s/ROC_statistics_%sdeg_P_%sdeg_%s_2RF.npz' % (self._path_RF_subregion, resolution, resolution, self._region_name), 
                 fpr=fpr, tpr=tpr, auc=roc_auc, thresholds=thresholds)

        return

    def score_feature_importance(self):
        """
        Save the feature importance
    
        """

        self.fit_RF()
        importances = self.reg.feature_importances_
        indices = np.argsort(importances)    
        np.savez('%s/feature_importance_%sdeg_P_%sdeg_%s_2RF.npz' % (self._path_RF_subregion, self._res_coarse ,self._res_coarse, self._region_name), importance=importances, rank=indices)
        feature_num = importances.shape[0]
        indices = indices[::-1]
        covariate_name = list(self.features_land_df.columns.values)
        std = np.std([tree.feature_importances_ for tree in self.reg.estimators_], axis=0)
        covariate_name_sort = [covariate_name[indices[i]] for i in range(feature_num)]
        print covariate_name_sort

        ### Save feature importance for the 2nd RF
        if self._seperateRF == True:
            importances_ext = self.reg_ext.feature_importances_
            indices_ext = np.argsort(importances_ext)    
            np.savez('%s/feature_importance_%sdeg_P_%sdeg_%s_2RF_ext.npz' % (self._path_RF_subregion, self._res_coarse ,self._res_coarse, self._region_name), importance=importances_ext, rank=indices_ext)
            indices_ext = indices_ext[::-1]
            covariate_name_sort_ext = [covariate_name[indices_ext[i]] for i in range(feature_num)]
            print covariate_name_sort_ext

    
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])
        ax.bar(range(feature_num), importances[indices], color="r", yerr=std[indices], align="center")
        ax.set_xlim([-1, feature_num])
        ax.set_xticks(range(feature_num))
        ax.set_xticklabels(covariate_name_sort, rotation=90)
        ax.set_title("Feature importances")

        return

    def score_minSemivariance(self, pre, resolution=None):
        """
        Use this function to output the first point of semivariogram 

        Args:
            :pre (array): 3d precipitation
    
        """

        resolution = resolution or self._res_coarse
        gamma = np.array([self.compute_variogram(pre, i)[1][0] for i in range(self._ntime)])
        np.savez('%s/minSemivariance_downscaled_%s_%sdeg_P_%sdeg_2RF.npz' % (self._path_RF_subregion, self._region_name, resolution, resolution), gamma=gamma) 

        return

    def cmap_customized(self):
        """
        Defined customized color table

        """

        matplotlib.rcParams['pdf.fonttype'] = 42
        cpalette = np.loadtxt('./WhiteBlueGreenYellowRed.rgb',skiprows=2)/255.
        cmap = colors.ListedColormap(cpalette, 256)
        cmap.set_bad('0.8') 
 
        return cmap

    def imshow_prec_obs(self, obs, itime=0, vmax=None):
        """
        Plot precipitation using customized color table

        Args:
            :obs (array): observed precipitation 
            :itime (int): ith time step
            :vmax (float): max value for colorbar 
    
        """

        # Show the spatial pattern
        cmap = self.cmap_customized()
        plt.figure()
        M = Basemap(resolution='l', llcrnrlat=self._minlat+0.1, urcrnrlat=self._maxlat+0.1, llcrnrlon=self._minlon+0.1, urcrnrlon=self._maxlon+0.1)
        M.imshow(np.ma.masked_equal(obs \
                   .reshape(-1, self._nlat_fine, self._nlon_fine)[itime], -9.99e+08), 
                   cmap=cmap, 
                   interpolation='nearest', 
                   vmin=0, 
                   vmax=vmax) 
        M.drawcoastlines()
        M.drawstates()
        M.colorbar()
        plt.title('Observed')
        # plt.savefig('../../Figures/Animation/%s_SEUS_adjacent_0.5deg_bi-linear_%s.png' % (title, i), format='PNG')
        plt.show()

    def imshow_prec_pre(self, pre, itime=0, vmax=None, vmin=0, title=None):
        """
        Plot precipitation using customized color table

        Args:
            :pre (array): downscaled precipitation
            :itime (int): ith time step
            :vmax (float): max value for colorbar 
    
        """

        # Show the spatial pattern
        cmap = self.cmap_customized()
        plt.figure()
        M = Basemap(resolution='l', llcrnrlat=self._minlat, urcrnrlat=self._maxlat, llcrnrlon=self._minlon, urcrnrlon=self._maxlon)
        M.imshow(np.ma.masked_equal(pre
                   .reshape(-1, self._nlat_fine, self._nlon_fine)[itime], -9.99e+08), 
                   cmap=cmap, 
                   interpolation='nearest', 
                   vmin=vmin, 
                   vmax=vmax) 
        M.drawcoastlines()
        M.drawstates()
        M.colorbar()
        plt.title('%s' %(title))
        # plt.savefig('../../Figures/Animation/%s_SEUS_adjacent_0.5deg_bi-linear_%s.png' % (title, i), format='PNG')
        plt.show()

    def plot_R2_RMSE(self):
        """
        Plot R2 and RMSE in the same figure 
    
        """

        resolution = [0.25, 0.5, 1]

        R2 = np.array([0.799, 0.668, 0.501, 0.858, 0.747, 0.549])      # NWUS
        RMSE = np.array([0.117, 0.150, 0.184, 0.098, 0.131, 0.175])    # NWUS

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 6))

        # Plot RMSE
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.xaxis.tick_bottom()
        ax1.yaxis.tick_left()
        ax1.plot(RMSE[:3], '-p', color='#e34a33', linewidth=5, markersize=12, alpha=0.9, label='1RF')
        ax1.plot(RMSE[3:], '-p', color='#31a354', linewidth=5, markersize=12, alpha=0.9)
        ax1.set_ylim([0.8*RMSE.min(), 1.1*RMSE.max()])
        ax1.set_xlim([-0.5, 2.5])
        leg = ax1.legend(loc=4, prop={'size':15})
        leg.get_frame().set_linewidth(0.0)
        ax1.set_xticks([0, 1, 2])
        ax1.set_xticklabels(['0.25$^\circ$', '0.5$^\circ$', '1$^\circ$'], fontsize=15)
        ax1.tick_params(axis='y', labelsize=15)
        fig.text(0.15, 0.9, "RMSE", fontsize=20)

        # Plot R2
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.xaxis.tick_bottom()
        ax2.yaxis.tick_right()
        ax2.plot(R2[:3], '-p', color='#e34a33', linewidth=5, markersize=12, alpha=0.9)
        ax2.plot(R2[3:], '-p', color='#31a354', linewidth=5, markersize=12, alpha=0.9, label='2RF')
        ax2.set_ylim([0.8*R2.min(), 1.1*R2.max()])
        ax2.set_xlim([-0.5, 2.5])
        leg = ax2.legend(loc=3, prop={'size':15})
        leg.get_frame().set_linewidth(0.0)
        ax2.set_xticks([0, 1, 2])
        ax2.set_xticklabels(['0.25$^\circ$', '0.5$^\circ$', '1$^\circ$'], fontsize=15)
        #ax2.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
        #ax2.set_yticklabels([0.5, 0.6, 0.7, 0.8, 0.9], fontsize=15)
        ax2.tick_params(axis='y', labelsize=15)
        fig.text(0.82, 0.9, "R2", fontsize=20)

        fig.tight_layout()
        plt.savefig('../../Figures/RF/R2_RMSE_%s.pdf' % (self._region_name), format='PDF')
        plt.savefig('../../Figures/RF/R2_RMSE_%s.eps' % (self._region_name), format='EPS')
        plt.show()

    def plot_treeEns(self, xxx, stime=0, etime=None):
        """
        Plot precipitation ensembles

        Args:
            :prec_df (df): precipitation dataframe
            :stime (int): start time step
            :etime (int): end time step
    
        """

        # !!!Need to be modified
        data_path = '../../Data/Output/RF/'

        ### Read datasets
        prec_fine = np.fromfile('%s/apcpsfc_2011_JJA_SEUS.bin' % (data_path),'float32').reshape(-1, nlat_fine, nlon_fine)[:ntime]
        tree_image = np.fromfile('%s/tree_hour_image_LargeMeteo_1deg_P_1deg.bin' % (data_path)).reshape(nTree, ntime, nlat_fine, nlon_fine)
        obs_hour = np.ma.masked_equal(prec_fine,-9.99e+08).mean(-1).mean(-1)
        tree_hour = np.ma.masked_equal(tree_image,-9.99e+08).mean(-1).mean(-1).data
        stats = cbook.boxplot_stats(tree_hour)

        fig, ax = plt.subplots(figsize=(10,5))
        plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.1)
        ax.bxp(stats[sTime:eTime])
        plt.plot(obs_hour[sTime:eTime],linewidth=2.5,color='r')
        plt.xticks([])
        plt.xlabel('Time (%s-%s)'%(sTime, eTime))
        plt.title('Domain Averaged Prep')
        plt.show()

    def plot_settings_imp(self):
        """
        Plot settings for feature importance
    
        """

        imp_ref = np.load('%s/feature_importance_0.25deg_P_0.25deg_%s_1RF.npz' % (self._path_RF_subregion, self._region_name))
        self.sorted_idx = imp_ref['rank']
        print self.sorted_idx

        self.pos = np.arange(self.sorted_idx.shape[0]) + 0.5
        self.pos = self.pos*2

        return

    def plot_feature_importance_1RF(self):
        """
        Plot feature importance for 1RF
    
        """

        resolution = [0.25, 0.5, 1]

        self.plot_settings_imp()
        fig = plt.figure(figsize=(4,8))
        ax_size = [0.1, 0.1, 0.8, 0.8]
        ax_left = fig.add_axes(ax_size)
        ax_left.spines['bottom'].set_visible(False)
        ax_left.spines['left'].set_visible(False)
        ax_left.xaxis.tick_top()
        ax_left.yaxis.set_ticks_position('none')
        ax_left.xaxis.set_label_position('top')
        ax_left.set_yticklabels([])

        for i, iRes in enumerate(resolution):
            feature_importance = np.load('%s/feature_importance_%sdeg_P_%sdeg_%s_1RF.npz' % (self._path_RF_subregion, iRes, iRes, self._region_name))
            importance = feature_importance['importance']
            for j in range(len(importance)):
                p = patches.Rectangle((0, j+(1-width[i])/2.0), importance[self.sorted_idx[j]], width[i], fill=True, transform=ax_left.transData, lw=0, facecolor=colors_1RF[i])
                ax_left.add_patch(p)

        plt.xticks(np.arange(0, 0.85, 0.2), np.arange(0, 0.85, 0.2))
        plt.xlim([0.85, 0])
        plt.ylim([-1, 21.5])
        plt.show()

        return

    def plot_feature_importance_2RF(self):
        """
        Plot feature importance for 2RF
    
        """

        resolution = [0.25, 0.5, 1]

        self.plot_settings_imp()
        fig = plt.figure(figsize=(4,8))
        ax_size = [0.1, 0.1, 0.8, 0.8]
        ax = fig.add_axes(ax_size)
        for i, iRes in enumerate(resolution):
            feature_importance = np.load('%s/feature_importance_%sdeg_P_%sdeg_%s_2RF.npz' % (self._path_RF_subregion, iRes, iRes, self._region_name))
            importance = feature_importance['importance']
            ax.barh(self.pos, importance[self.sorted_idx], width[i], color=colors_2RF[i], align='center', linewidth=0, alpha=1)
        plt.xlim([0, 0.85])
        plt.ylim([-1, 43])
        plt.show()

        fig = plt.figure(figsize=(4,8))
        ax_size = [0.1, 0.1, 0.8, 0.8]
        ax = fig.add_axes(ax_size)
        for i, iRes in enumerate(resolution):
            feature_importance = np.load('%s/feature_importance_%sdeg_P_%sdeg_%s_2RF_ext.npz' % (self._path_RF_subregion, iRes, iRes, self._region_name))
            importance = feature_importance['importance']
            ax.barh(self.pos, importance[self.sorted_idx], width[i], color=colors_2RF[i], align='center', linewidth=0, alpha=1)
        plt.xlim([0, 0.85])
        plt.ylim([-1, 43])
        plt.show()

        return

    def plot_feature_importance_1RF_2RF(self):
        """
        Plot feature importance for 1RF, 2RF and 2RF_extreme together
    
        """

        self.plot_settings_imp()
        resolution = [0.25, 0.5, 1]

        matplotlib.rc('grid', color='white')
        matplotlib.rc('grid', linewidth=1)

        fig = plt.figure(figsize=(12,7), facecolor='white')

        # Plot feature importance for 1RF
        axes_left  = fig.add_axes([0.01, 0.01, 0.25, 0.92])

        # Keep only top and right spines
        axes_left.spines['left'].set_color('none')
        axes_left.spines['right'].set_zorder(10)
        axes_left.spines['bottom'].set_color('none')
        axes_left.xaxis.set_ticks_position('top')
        axes_left.yaxis.set_ticks_position('none')
        axes_left.set_yticklabels([])

        for i, iRes in enumerate(resolution):
            feature_importance = np.load('%s/feature_importance_%sdeg_P_%sdeg_%s_1RF.npz' % (self._path_RF_subregion, iRes, iRes, self._region_name))
            importance = feature_importance['importance']
            for j in range(len(importance)):
                p = patches.Rectangle((0, j+(1-width[i])/2.0), importance[self.sorted_idx[j]], width[i], fill=True, transform=axes_left.transData, lw=0, facecolor=colors_1RF[i])
                axes_left.add_patch(p)

        axes_left.set_xticks(np.arange(0.2, 0.85, 0.2))
        axes_left.set_xticklabels(np.arange(0.2, 0.85, 0.2))
        axes_left.set_xlim([0.85, 0])
        axes_left.set_ylim([-0.5, 21.5])
        axes_left.grid()

        arrowprops = dict(arrowstyle="-", connectionstyle="angle,angleA=0,angleB=90,rad=0")
        axes_left.annotate('0.25$^\circ$', xy=(.9*0.8, 20.5), xycoords='data', horizontalalignment='right', fontsize= 18, xytext=(0.25, 0.6), textcoords='axes fraction', arrowprops=arrowprops)
        axes_left.annotate('0.5$^\circ$', xy=(.9*0.5, 20.5), xycoords='data', horizontalalignment='right', fontsize= 18, xytext=(0.55, 0.6), textcoords='axes fraction', arrowprops=arrowprops)
        axes_left.annotate('1$^\circ$', xy=(.9*0.2, 20.5), xycoords='data', horizontalalignment='right', fontsize= 18, xytext=(0.85, 0.6), textcoords='axes fraction', arrowprops=arrowprops)

        # Plot feature importance for 2RF (normal)
        axes_right  = fig.add_axes([0.45, 0.01, 0.25, 0.92])

        # Keep only top and left spines
        axes_right.spines['right'].set_color('none')
        axes_right.spines['left'].set_zorder(10)
        axes_right.spines['bottom'].set_color('none')
        axes_right.xaxis.set_ticks_position('top')
        axes_right.yaxis.set_ticks_position('none')
        axes_right.set_yticklabels([])

        for i, iRes in enumerate(resolution):
            feature_importance = np.load('%s/feature_importance_%sdeg_P_%sdeg_%s_2RF.npz' % (self._path_RF_subregion, iRes, iRes, self._region_name))
            importance = feature_importance['importance']
            for j in range(len(importance)):
                p = patches.Rectangle((0, j+(1-width[i])/2.0), importance[self.sorted_idx[j]], width[i], fill=True, transform=axes_right.transData, lw=0, facecolor=colors_2RF[i])
                axes_right.add_patch(p)

        axes_right.set_xticks(np.arange(0.2, 0.85, 0.2))
        axes_right.set_xticklabels(np.arange(0.2, 0.85, 0.2))
        axes_right.set_xlim([0, 0.85])
        axes_right.set_ylim([-0.5, 21.5])
        axes_right.grid()

        # Plot feature importance for 2RF (extreme)
        axes_right_ext  = fig.add_axes([0.72, 0.01, 0.25, 0.92])

        # Keep only top and left spines
        axes_right_ext.spines['right'].set_color('none')
        axes_right_ext.spines['left'].set_zorder(10)
        axes_right_ext.spines['bottom'].set_color('none')
        axes_right_ext.xaxis.set_ticks_position('top')
        axes_right_ext.yaxis.set_ticks_position('none')
        axes_right_ext.set_yticklabels([])

        for i, iRes in enumerate(resolution):
            feature_importance = np.load('%s/feature_importance_%sdeg_P_%sdeg_%s_2RF_ext.npz' % (self._path_RF_subregion, iRes, iRes, self._region_name))
            importance = feature_importance['importance']
            for j in range(len(importance)):
                p = patches.Rectangle((0, j+(1-width[i])/2.0), importance[self.sorted_idx[j]], width[i], fill=True, transform=axes_right_ext.transData, lw=0, facecolor=colors_2RF[i])
                axes_right_ext.add_patch(p)

        axes_right_ext.set_xticks(np.arange(0.2, 0.85, 0.2))
        axes_right_ext.set_xticklabels(np.arange(0.2, 0.85, 0.2))
        axes_right_ext.set_xlim([0, 0.85])
        axes_right_ext.set_ylim([-0.5, 21.5])
        axes_right_ext.grid()
        axes_right.annotate('1$^\circ$', xy=(.9*0.2, 20.5), xycoords='data', horizontalalignment='right', fontsize= 18, xytext=(0.25, 0.6), textcoords='axes fraction', arrowprops=arrowprops)
        axes_right.annotate('0.5$^\circ$', xy=(.9*0.5, 20.5), xycoords='data', horizontalalignment='right', fontsize= 18, xytext=(0.6, 0.6), textcoords='axes fraction', arrowprops=arrowprops)
        axes_right.annotate('0.25$^\circ$', xy=(.9*0.8, 20.5), xycoords='data', horizontalalignment='right', fontsize= 18, xytext=(0.95, 0.6), textcoords='axes fraction', arrowprops=arrowprops)

        # Y axis labels
        fea_label = ['Veg type', 'Texture', 'Elevation (mean)', 'DOY', 'Lon', 'Slope', 'Pressure', 'Elevation (std)', 'Aspect', 'Lat', 'Meridional wind', 'Humidity', 'CAPE', 'Zonal wind', 'Temperature', 'Prec (up)', 'Prec (left)', 'Prec (right)', 'Prec (down)', 'Distance', 'Prec (central)']    # SEUS

        for i in range(21):
            x1, y1 = axes_left.transData.transform_point((0, i+.5))
            x2, y2 = axes_right.transData.transform_point((0, i+.5))
            x, y = fig.transFigure.inverted().transform_point( ((x1+x2)/2,y1) )
            plt.text(x, y, fea_label[i], transform=fig.transFigure, fontsize=15, horizontalalignment='center', verticalalignment='center')

        #fig.tight_layout()

        axes_left.text(0.4, 0.27, "1RF", fontsize=20, fontweight='bold', horizontalalignment='left', verticalalignment='top', transform=axes_left.transAxes)
        axes_right.text(0.5, 0.27, "2RF \n (Moderate)", fontsize=20, fontweight='bold', horizontalalignment='center', verticalalignment='top', transform=axes_right.transAxes) 
        axes_right_ext.text(0.5, 0.27, "2RF \n (Extreme)", fontsize=20, fontweight='bold', horizontalalignment='center', verticalalignment='top', transform=axes_right_ext.transAxes)

        plt.savefig('../../Figures/RF/feature_importance_%s.pdf' % (self._region_name), format='PDF')
        plt.savefig('../../Figures/RF/feature_importance_%s.eps' % (self._region_name), format='EPS')

        return

    def plot_QQ(self):
        """
        Plot quantile quantile plot 
    
        """

        resolution = [0.25, 0.5, 1]
        fig = plt.figure(figsize=(6, 6))

        for i, iRes in enumerate(resolution):
            qq_obs_1RF = np.fromfile('%s/quantiles_obsmask_LargeMeteo_%sdeg_P_%sdeg_NWUS_1RF.bin' % (self._path_RF_subregion, iRes, iRes), 'float32')
            qq_obs_2RF = np.fromfile('%s/quantiles_obsmask_LargeMeteo_%sdeg_P_%sdeg_NWUS_2RF.bin' % (self._path_RF_subregion, iRes, iRes), 'float32')
            qq_pred_1RF = np.fromfile('%s/quantiles_downscaled_LargeMeteo_%sdeg_P_%sdeg_NWUS_1RF.bin' % (self._path_RF_subregion, iRes, iRes), 'float64')
            qq_pred_2RF = np.fromfile('%s/quantiles_downscaled_LargeMeteo_%sdeg_P_%sdeg_NWUS_2RF.bin' % (self._path_RF_subregion, iRes, iRes), 'float64')
            plt.scatter(qq_pred_1RF, qq_obs_1RF, color=colors_1RF[i], alpha=1, label='1RF_%s$^\circ$' % (iRes))
            plt.scatter(qq_pred_2RF, qq_obs_2RF, color=colors_2RF[i], alpha=1, label='2RF_%s$^\circ$' % (iRes))

        plt.rc('font', family='Arial')
        plt.plot([-10, 1.1*qq_obs_1RF.max()], [-10, 1.1*qq_obs_1RF.max()], 'k--', linewidth=1.5)
        plt.xlim([-10, 1.1*qq_obs_1RF.max()])
        plt.ylim([-10, 1.1*qq_obs_1RF.max()])
        leg = plt.legend(loc=4, prop={'size':15})
        leg.get_frame().set_linewidth(0.0)
        plt.xlabel('Downscaled precipitation [mm]', size=18)
        plt.ylabel('Observed precipitation [mm]', size=18)
        fig.tight_layout()
        plt.savefig('../../Figures/RF/QQ_plot_%s.png' % (self._region_name), format='PNG')
        #plt.savefig('../../Figures/RF/QQ_plot_%s.pdf' % (self._region_name), format='PDF')
        #plt.savefig('../../Figures/RF/QQ_plot_%s.eps' % (self._region_name), format='EPS')
        plt.show()

        return

    def plot_ROC(self):
        """
        Plot ROC (Receiver Operating Characteristic) plot 
    
        """

        resolution = [0.25, 0.5, 1]
        colors = ['#f03b20', '#feb24c', '#c51b8a']        # For upscaled atmospheric covariates
        # colors = ['#31a354', '#addd8e', '#67a9cf']      # For 0.125 deg atmospheric covariates
        fig = plt.figure(figsize=(6, 6))

        for i, iRes in enumerate(resolution):
            ROC_metrics_1 = np.load('%s/ROC_statistics_%sdeg_P_%sdeg_%s_2RF.npz' % (self._path_RF_subregion, iRes, iRes, self._region_name))
            #ROC_metrics_2 = np.load('%s/ROC_statistics_LargeMeteo_0.125deg_P_%s.npz' % (data_path, iRes))
            plt.plot(ROC_metrics_1['fpr'], ROC_metrics_1['tpr'], colors[i], linewidth=2.5, label='%s deg (AUC = %0.2f)' % (iRes, ROC_metrics_1['auc']))
            #plt.plot(ROC_metrics_2['fpr'], ROC_metrics_2['tpr'], colors_2[i], linewidth=2.5, label='%s (AUC = %0.2f)' % (iRes, ROC_metrics_2['auc']))

        plt.rc('font', family='Arial')
        plt.xlim([-0.05, 0.25])
        plt.ylim([0.4, 1.05])
        plt.legend(loc=4, prop={'size':15})
        plt.xlabel('False Positive Ratio', size=20)
        plt.ylabel('True Positive Ratio', size=20)
        plt.title('ROC Curve', size=25)
        plt.show()

        return

    def plot_semivariance_scatter(self):
        """
        Scatter plot of the minimum semivariance from the semi-variogram 
    
        """

        resolution = [0.25, 0.5, 1]
        fig = plt.figure(figsize=(6, 6))

        gamma_obs = np.load('%s/minSemivariance_obs_NWUS.npz' % (self._path_RF_subregion))
        for i, iRes in enumerate(resolution):
            gamma_1RF = np.load('%s/minSemivariance_downscaled_NWUS_%sdeg_P_%sdeg_1RF.npz' % (self._path_RF_subregion, iRes, iRes))
            gamma_2RF = np.load('%s/minSemivariance_downscaled_NWUS_%sdeg_P_%sdeg_2RF.npz' % (self._path_RF_subregion, iRes, iRes))
            plt.scatter(gamma_obs['gamma'], gamma_1RF['gamma'], marker='^', color=colors_1RF[i], label='1RF_%s$^\circ$' % (iRes), s=25)
            plt.scatter(gamma_obs['gamma'], gamma_2RF['gamma'], marker='p', color=colors_2RF[i], label='2RF_%s$^\circ$' % (iRes), s=25)

        plt.plot([0,1],[0,1], '--k', linewidth=1.5)
        plt.rc('font', family='Arial')
        plt.xlim([0.05, 1])
        plt.ylim([0.05, 1])
        leg = plt.legend(loc=2, prop={'size':15})
        leg.get_frame().set_linewidth(0.0)
        plt.xlabel('Observed semivariance (lag 1) [-]', size=18)
        plt.ylabel('Downscaled semivariance (lag 1) [-]', size=18)
        fig.tight_layout()
        plt.savefig('../../Figures/RF/semivariance_scatter_%s.pdf' % (self._region_name), format='PDF')
        plt.savefig('../../Figures/RF/semivariance_scatter_%s.eps' % (self._region_name), format='EPS')
        plt.show()

        return

    def plot_semivariance_ts(self, sTime, eTime):
        """
        Plot time series of the minimum semivariance from the semi-variogram
    
        """

        resolution = [0.25, 0.5, 1]
        gamma_obs = np.load('%s/minSemivariance_obs_NWUS.npz' % (self._path_RF_subregion))

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 6))
        for i, iRes in enumerate(resolution):
            gamma_1RF = np.load('%s/minSemivariance_downscaled_NWUS_%sdeg_P_%sdeg_1RF.npz' % (self._path_RF_subregion, iRes, iRes))
            gamma_2RF = np.load('%s/minSemivariance_downscaled_NWUS_%sdeg_P_%sdeg_2RF.npz' % (self._path_RF_subregion, iRes, iRes))
            axes[i].plot(gamma_obs['gamma'][sTime:eTime], color='k', linewidth=2, label='Obs')
            axes[i].plot(gamma_1RF['gamma'][sTime:eTime], color='#e34a33', alpha=0.9, linewidth=2, label='1RF')
            axes[i].plot(gamma_2RF['gamma'][sTime:eTime], color='#31a354', alpha=0.9, linewidth=2, label='2RF')
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['bottom'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].xaxis.tick_bottom()
            axes[i].yaxis.tick_left()
            axes[i].set_xticks([])
            axes[i].set_xticklabels([])
            axes[i].set_yticks(np.arange(0.2, 1.2, 0.2))
            axes[i].set_yticklabels(np.arange(0.2, 1.2, 0.2), fontsize=15)
            axes[i].text(0.03, 0.85, "%s$^\circ$" % (iRes), fontsize=18, horizontalalignment='left', verticalalignment='center', transform=axes[i].transAxes)
            axes[i].set_axis_bgcolor('#E0E0E0')
            
        leg = axes[0].legend(loc=1, prop={'size':15})
        leg.get_frame().set_linewidth(0.0)
        leg.get_frame().set_color('#E0E0E0')
        axes[1].set_ylabel('Semivariance [-]', fontsize=18)
        axes[2].spines['bottom'].set_visible(True)
        axes[2].set_xticks(np.arange(0, eTime-sTime+1, 20))
        axes[2].set_xticklabels(np.arange(sTime, eTime+1, 20), fontsize=15)
        axes[2].set_xlabel('Time step [-]', fontsize=18)
        plt.subplots_adjust(hspace=0.05, left=0.06, right=0.97, bottom=0.12, top=0.96)

        plt.savefig('../../Figures/RF/semivariance_ts_%s.pdf' % (self._region_name), format='PDF')
        plt.savefig('../../Figures/RF/semivariance_ts_%s.eps' % (self._region_name), format='EPS')
        plt.show()

    def plot_spatial_corr(nlon, nlat, data):
        """
        Plot the spatial correlation using R's 'ncf' package

        Input:  nlon: lon number
                nlat: lat number
                data: 1-d array
        """

        ##### Load R packages
        importr('ncf')

        lon = r("lon <- expand.grid(1:%d, 1:%d)[,1]" % (nlon, nlat))
        lat = r("lat <- expand.grid(1:%d, 1:%d)[,2]" % (nlon, nlat))
        lon = np.array(lon)
        lat = np.array(lat)

        ind = data != -9.99e+08
        data = data[ind]
        lon = lon[ind]
        lat = lat[ind]

        ##### Convert numpy to R format
        data = FloatVector(data)
        lon = FloatVector(lon)
        lat = FloatVector(lat)

        globalenv['data'] = data
        globalenv['lon'] = lon
        globalenv['lat'] = lat
        fit = r("spline.correlog(x=lon, y=lat, z=data, resamp=5)")

        ##### Convert R object to Python Dictionary
        fit = com.convert_robj(fit)

        ##### Plot
        plt.figure()
        plt.plot(fit['real']['predicted']['x'], fit['real']['predicted']['y'])
        plt.plot(fit['real']['predicted']['x'], fit['boot']['boot.summary']['predicted']['y'].loc[['0']].values.squeeze())  # Lower boundary
        plt.plot(fit['real']['predicted']['x'], fit['boot']['boot.summary']['predicted']['y'].loc[['1']].values.squeeze())  # Upper boundary
        plt.show()

    def compute_frac_area_intensity(data_fine, ngrid, nlat_coarse, nlon_coarse):
        """
        Compute the precipitaton fraction area and precipitation intensity averaged over the large grid box

        Input:  data_fine: 2-d array; size=(nlat_fine, nlon_fine)
                ngrid: number of small grid cells within large grid cell
                nlat_coarse: lat number of coarse resolution
                nlon_coarse: lon number of coarse resolution
        """
        data_fine_group = np.array([data_fine[i*ngrid:(i+1)*ngrid, j*ngrid:(j+1)*ngrid] for i in range(nlat_coarse) for j in range(nlon_coarse)])
        data_fine_group_mask = np.ma.masked_equal(data_fine_group, -9.99e+08)
        data_fine_group_label = (data_fine_group_mask>0).astype('float')
        prec_frac_area = data_fine_group_label.mean(-1).mean(-1)
        prec_intensity = data_fine_group_mask.mean(-1).mean(-1)

        return prec_frac_area, prec_intensity

    def pred_ints(model, X, percentile=95):
        """
        http://blog.datadive.net/prediction-intervals-for-random-forests/

        Reference: Meinshausen (2006), Quantile Regression Forests
        """
        tree_num = len(model.estimators_)
        preds = np.array([model.estimators_[i].predict(X) for i in range(tree_num)])
        err_down = np.percentile(preds, (100-percentile)/2., axis=0)
        err_up = np.percentile(preds, 100-(100-percentile)/2., axis=0)

        return err_down, err_up, preds

    def print_info_to_command_line(line):
        print "\n"
        print "#######################################################################################"
        print "%s" % line
        print "#######################################################################################"
        print "\n"

        return
        

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size
        
    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols))

def disaggregate(inArr, gridNo):
    out = np.repeat(np.repeat(inArr, gridNo, axis=1), gridNo, axis=0)
    return out

def upscale(arr, nrows, ncols, nlat_coarse, nlon_coarse):
    """
    Return an upscaled array of shape (ngrid_Up, ngrid_Up)
    
    Note: Similar to the blockshaped function
    """

    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols)).mean(-1).mean(-1).reshape(nlat_coarse, nlon_coarse)

