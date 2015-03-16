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

def upscale(arr, nrows, ncols, ngrid_Up):
        """
        Return an upscaled array of shape (ngrid_Up, ngrid_Up)

	Note: Similar to the blockshaped function
        """

        h, w = arr.shape
        return (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols)).mean(-1).mean(-1).reshape(ngrid_Up, ngrid_Up)

