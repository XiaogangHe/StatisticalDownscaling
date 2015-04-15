#!/usr/bin/env python

import	numpy			as	np
import	matplotlib.pyplot	as	plt
from 	netCDF4			import	Dataset

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

