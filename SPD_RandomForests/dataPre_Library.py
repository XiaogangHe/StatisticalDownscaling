#!/usr/bin/env python

import	numpy                   as      np
import	matplotlib.pyplot       as      plt
from 	netCDF4                 import	Dataset
import  pandas.rpy.common       as      com
from    rpy2.robjects           import  r
from    rpy2.robjects.packages  import  importr
from    rpy2.robjects.vectors   import  FloatVector
from    rpy2.robjects           import  globalenv
import  rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

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

def extend_array_boundary(inArr):
    """
    Add the boundary to the original array 

    Change shape from (x,y) to (x+2,y+2)
    """

    add_row = np.r_[[inArr[0]], inArr, [inArr[-1]]]
    add_row_col = np.c_[add_row[:,0], add_row, add_row[:,-1]]
    return add_row_col

def get_adjacent_grids(extendArr, rand_row, rand_col):
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
    return err_down, err_up

