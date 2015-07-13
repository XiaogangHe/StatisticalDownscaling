#!/usr/bin/env python

from 	netCDF4                 import	Dataset
from    pandas                  import  DataFrame
import	numpy                   as      np
import	matplotlib.pyplot       as      plt
import  pandas                  as      pd
import  pandas.rpy.common       as      com
from    rpy2.robjects           import  r
from    rpy2.robjects.packages  import  importr
from    rpy2.robjects.vectors   import  FloatVector
from    rpy2.robjects           import  globalenv
import  rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

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
        self._minlat = region_info['minlat']
        self._minlon = region_info['minlon']
        self._maxlat = region_info['maxlat']
        self._maxlon = region_info['maxlon']
        self._nlat_fine = region_info['nlat_fine']
        self._nlon_fine = region_info['nlon_fine']
        self._res_fine = region_info['res_fine']
        self._stime = date_info['stime']
        self._ntime = date_info['ntime']
        self._path = data_info['path']
        self._rand_row_num = RF_config['rand_row_num']
        self._rand_col_num = RF_config['rand_col_num']
        self._ntree = RF_config['ntree']
        self._njob = RF_config['njob']
        self._features_static = features_name['static']
        self._features_dynamic = features_name['dynamic']

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
        DOYs = np.array([np.ones((self._nlat_fine, self._nlon_fine))*DOY[i] for i in range(self._ntime)])
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

    def prepare_covariates(self):
        """
        Prepare precipitation observations
    
        """

        # Add static covariates
        self.features_dic = {static_feature: \
                np.fromfile('%s/%s_SEUS.bin' % (self._path, static_feature),'float32').reshape(self._nlat_fine, self._nlon_fine) \
                for static_feature in self._features_static} 

        # Add dynamic covariates
        features_dic_dynamic = {dynamic_feature: \
                np.fromfile('%s/%s_UpDown_0.5deg_2011_JJA_SEUS_bi-linear.bin' % \
                (self._path, dynamic_feature),'float32').reshape(-1, self._nlat_fine, self._nlon_fine)[:self._ntime] \
                for dynamic_feature in self._features_dynamic} 
        self.features_dic.update(features_dic_dynamic)

        # Add adjacent precipitation grid cells as covariates
        prec_UpDown = self.features_dic['prec']
        self.prec_UpDown_extend = np.array([self.extend_array_boundary(prec_UpDown[i]) for i in range(self._ntime)])
        self.features_dic['prec_disagg_l'] = self.prec_UpDown_extend[:, 1:-1, :-2]
        self.features_dic['prec_disagg_r'] = self.prec_UpDown_extend[:, 1:-1, 2:]
        self.features_dic['prec_disagg_u'] = self.prec_UpDown_extend[:, :-2, 1:-1]
        self.features_dic['prec_disagg_d'] = self.prec_UpDown_extend[:, 2:, 1:-1]

        # Add DOY (day of year) as covariates
        self.features_dic['DOY'] = self.get_DOYs()

        # Add lats & lons as covariates
        self.features_dic['lons'] = self.get_lons_lats()[0]
        self.features_dic['lats'] = self.get_lons_lats()[1]

        #return self.features_dic
        return 

    def prepare_prec_fine(self):
        """
        Prepare precipitation observations at fine resolution
    
        """
        file_name = 'apcpsfc_2011_JJA_SEUS.bin'
        self.prec_fine = np.fromfile('%s/%s' % (self._path, file_name),'float32').reshape(-1, self._nlat_fine, self._nlon_fine)[:self._ntime]
        return self.prec_fine

    def mask_out_ocean(self, covariates_df, response_df):
        """
        This is a class to use Random Forests for precipitation downscaling
    
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

        self.prepare_prec_fine()
        self.prepare_covariates()

        features_name_train = self.features_dic.keys()
        features_name_train.remove('prec')
        self.features_train_dic = {feature_train: [] for feature_train in features_name_train} 
        self.features_train_dic['prec_disagg_c'] = [] 
        self.prec_fine_train = [] 
        
        dynamic_feature_update = [name for name in self._features_dynamic if name != 'prec']
        dynamic_feature_update.extend(['DOY', 'lons', 'lats'])            
        self.prec_feature = ['prec_disagg_c', 'prec_disagg_l', 'prec_disagg_r', 'prec_disagg_u', 'prec_disagg_d']

        for i in range(self._ntime):
            ### Random choose grids
            rand_row_ind = np.random.choice(self._nlat_fine, self._rand_row_num, replace=False)
            rand_col_ind = np.random.choice(self._nlon_fine, self._rand_col_num, replace=False)

            ### Random sample fine precipitation
            self.prec_fine_train.append(self.prec_fine[i][np.ix_(rand_row_ind, rand_col_ind)])

            ### Random sample coarse precipitation with its adjacent grids
            for prec_ind, prec_name in enumerate(self.prec_feature):
                grid_loc = self.get_adjacent_grids(self.prec_UpDown_extend[i], rand_row_ind, rand_col_ind)[prec_ind]
                self.features_train_dic[prec_name].append(grid_loc)

            ### Random sample other variables 
            for static_feature in self._features_static:
                self.features_train_dic[static_feature].append(self.features_dic[static_feature][np.ix_(rand_row_ind, rand_col_ind)])
            
            for dynamic_feature in dynamic_feature_update:
                self.features_train_dic[dynamic_feature].append(self.features_dic[dynamic_feature][i][np.ix_(rand_row_ind, rand_col_ind)])

        ### Create dataframe for features
        self.features_name_all = self.features_train_dic.keys()
        features_train_df = DataFrame({feature_all: np.array(self.features_train_dic[feature_all]).reshape(-1) for feature_all in self.features_name_all}) 
        features_df = DataFrame({static_feature: np.array([self.features_dic[static_feature].tolist()]*self._ntime).reshape(-1) for static_feature in self._features_static}) 
        for dynamic_feature in dynamic_feature_update + self.prec_feature:
            if dynamic_feature == 'prec_disagg_c':
                features_df[dynamic_feature] = self.features_dic['prec'].reshape(-1)
            else:
                features_df[dynamic_feature] = self.features_dic[dynamic_feature].reshape(-1)

        ### Create dataframe for precipitation
        prec_fine_train_df = DataFrame({'prec_fine': np.array(self.prec_fine_train).reshape(-1)})
        prec_fine_df = DataFrame({'prec_fine': self.prec_fine.reshape(-1)})

        ### Mask out ocean grid cells
        self.features_train_land_df, self.prec_fine_train_land_df = self.mask_out_ocean(features_train_df, prec_fine_train_df)
        self.features_land_df, self.prec_fine_land_df = self.mask_out_ocean(features_df, prec_fine_df)
        self.features_land_df = self.features_land_df.sort_index(axis=1)

        # return self.features_train_land_df, self.prec_fine_train_land_df, self.features_land_df, self.prec_fine_land_df 
        return 

    def ensemble_mean(self):
        """
        Use random forests to train and downscale coarse resolution precipitation
    
        """
        from sklearn.ensemble import RandomForestRegressor

        prec_prediction_df = DataFrame({'prec_fine': np.array([-9.99e+08]*self._nlat_fine*self._nlon_fine*self._ntime)})
        self.prepare_training_data()

        reg = RandomForestRegressor(n_estimators=self._ntree, bootstrap=True, oob_score=True, n_jobs=self._njob)
        reg.fit(self.features_train_land_df, np.ravel(self.prec_fine_train_land_df))
        prec_pre_all = reg.predict(self.features_land_df)
        prec_prediction_df['prec_fine'][self.features_land_df.index] = prec_pre_all.astype('float32')

        return prec_prediction_df

    def plot_feature_importance(reg, feature_num, feature_name):
        importances = reg.feature_importances_
        std = np.std([tree.feature_importances_ for tree in reg.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]    
        feature_name_sort = [feature_name[indices[i]] for i in range(feature_num)]
    
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])
        ax.bar(range(feature_num), importances[indices], color="r", yerr=std[indices], align="center")
        ax.set_xlim([-1, feature_num])
        ax.set_xticks(range(feature_num))
        ax.set_xticklabels(feature_name_sort, rotation=90)
        ax.set_title("Feature importances")

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

    def compute_variogram(nlon, nlat, data, psill=0, vrange=40, nugget=0):
        """
        Compute the semi-variogram using R's 'gstat' package

        Input:  nlon: lon number
                nlat: lat number
                data: 1-d array
        """

        ##### Load R packages
        r('library("gstat")')
        r('library("sp")')

        lon = r("lon <- expand.grid(1:%d, 1:%d)[,1]" % (nlon, nlat))
        lat = r("lat <- expand.grid(1:%d, 1:%d)[,2]" % (nlon, nlat))
        lon = np.array(lon)
        lat = np.array(lat)

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

