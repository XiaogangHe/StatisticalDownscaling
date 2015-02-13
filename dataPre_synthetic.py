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

def     disaggregate(inArr, gridNo):
        out = np.repeat(np.repeat(inArr,[gridNo,gridNo],axis=1),[gridNo,gridNo],axis=0)
        return out

##### Prepare the data for spTimer 
year1		= np.arange(1982,2009)
y2009		= np.repeat(np.array([2009]),10)
year		= np.append(np.repeat(year1,12),y2009)
nmon		= year.shape[0]
nyear		= 27
sLat		= 0
eLat		= 16
#eLat		= 81
sLon		= 0
eLon		= 16
#eLon		= 73

##### Read the data 

NLDAS		= Dataset('../Data/pcpNLDAS_3month.nc')
NLDAS_lat	= NLDAS.variables['lat'][sLat:eLat]
NLDAS_lon	= NLDAS.variables['lon'][sLon:eLon]
NLDAS_nlat	= NLDAS_lat.shape[0]
NLDAS_nlon	= NLDAS_lon.shape[0]
NLDAS_ngrid	= NLDAS_nlat*NLDAS_nlon	
NLDAS_prec	= NLDAS.variables['pcp'][sLat:eLat,sLon:eLon,:334]				# 334=27*12+10
nan_arr		= [np.nan]*NLDAS_nlon
NLDAS_prec2	= np.array([np.column_stack([NLDAS_prec[i],nan_arr,nan_arr]) for i in range(NLDAS_nlat)])
NLDAS_prec_YM	= NLDAS_prec2.reshape(NLDAS_nlat,NLDAS_nlon,-1,12)		# shape=(81,73,28,12)
NLDAS_prec_cli	= np.nanmean(NLDAS_prec_YM,axis=2)
NLDAS_prec_YM	= NLDAS_prec_YM.reshape(NLDAS_nlat,NLDAS_nlon,-1)[:,:,:-2]

sp_lon1			= np.repeat(NLDAS_lon,nmon).reshape(-1,nmon).tolist()
sp_lon			= np.array([[sp_lon1]*NLDAS_nlat]).squeeze().reshape(-1)
sp_lat			= np.repeat(NLDAS_lat,NLDAS_nlon*nmon)
sp_year			= np.array([[year.tolist()]*NLDAS_ngrid]).squeeze().reshape(-1)
print			NLDAS_prec_YM.shape
sp_prec_NLDAS_YM	= NLDAS_prec_YM.reshape(-1)
sp_prec_NLDAS_YM_UP	= np.array([blockshaped(NLDAS_prec_YM[:,:,i], 8, 8).mean(-1).mean(-1).reshape(2,2) for i in range(334)])
temp			= np.array([disaggregate(sp_prec_NLDAS_YM_UP[i],8) for i in range(334)])
sp_prec_NLDAS_YM_UpDown	= np.transpose(temp, (1, 2, 0)).reshape(-1)
print			sp_prec_NLDAS_YM_UpDown.shape

mon			= np.append(np.array([[np.arange(1,13).tolist()]*nyear]).squeeze().reshape(-1),np.arange(1,11))
sp_mon			= np.array([[mon.tolist()]*NLDAS_ngrid]).squeeze().reshape(-1)

sp_file			= np.vstack((sp_lon,sp_lat,sp_year,sp_mon,sp_prec_NLDAS_YM,sp_prec_NLDAS_YM_UpDown)).T
np.savetxt('test_syn.txt',sp_file,header='Longitude Latitude Year Month NLDAS_prec NLDAS_UpDown', fmt=['%5.3f','%5.3f','%g','%g','%5.3f','%5.3f'])

plt.figure()
plt.imshow(temp.mean(0),interpolation='nearest')
plt.title('NLDAS Upscale',fontsize=20)
plt.colorbar()

plt.figure()
plt.imshow(NLDAS_prec_YM.mean(-1))
plt.title('NLDAS',fontsize=20)
plt.colorbar()

plt.show()
