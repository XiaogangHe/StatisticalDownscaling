import	numpy			as	np
import	matplotlib.pyplot	as	plt
import	matplotlib.cm		as	cm
import	matplotlib
#from	matplotlib.pyplot	import	*

def blockshaped(arr, nrows, ncols):
	"""
	Return an array of shape (n, nrows, ncols) where
        n * nrows * ncols = arr.size
	
	If arr is a 2D array, the returned array should look like n subblocks with
	each subblock preserving the "physical" layout of arr.
	"""
	h, w = arr.shape
	return (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols))

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
	cdict = {
	        'red': [],
	        'green': [],
	        'blue': [],
	        'alpha': []
	    }
	
	# regular index to compute the colors
	reg_index = np.linspace(start, stop, 257)
	
	# shifted index to match the data
	shift_index = np.hstack([
		np.linspace(0.0, midpoint, 128, endpoint=False), 
		np.linspace(midpoint, 1.0, 129, endpoint=True)
	])

	for ri, si in zip(reg_index, shift_index):
	        r, g, b, a = cmap(ri)
	        cdict['red'].append((si, r, r))
		cdict['green'].append((si, g, g))
		cdict['blue'].append((si, b, b))
		cdict['alpha'].append((si, a, a))

	newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
	plt.register_cmap(cmap=newcmap)
	
	return newcmap

def	calMSESS(obs,pre):
	MSEc	= np.std(obs,axis=0)
	MSEf	= (pre-obs)**2
	MSEf	= MSEf.mean(0)
	MSESS	= 1-MSEf/MSEc/MSEc

	return	MSESS


sYear	= 1982
eYear	= 2009	# 2008 when preMon=12
nYear	= eYear-sYear+1
Year	= np.arange(sYear,eYear+1)
nlat	= 16
nlon	= 16
preMon	= 8
strMon	= 'Aug'

obs	= np.array([np.loadtxt('../Data/NLDAS_obs_%s%02d.txt'%(iyear,preMon)).reshape(nlat,nlon) for iyear in range(sYear,eYear+1)])
pre	= np.array([np.loadtxt('../Data/NLDAS_pre_%s%02d.txt'%(iyear,preMon)).reshape(nlat,nlon) for iyear in range(sYear,eYear+1)])
pre_Up	= np.array([np.loadtxt('../Data/NLDAS_pre_Up_%s%02d.txt'%(iyear,preMon)).reshape(nlat,nlon) for iyear in range(sYear,eYear+1)])
pre_Low	= np.array([np.loadtxt('../Data/NLDAS_pre_Low_%s%02d.txt'%(iyear,preMon)).reshape(nlat,nlon) for iyear in range(sYear,eYear+1)])
NMME	= blockshaped(pre.mean(0), 8, 8).mean(-1).mean(-1).reshape(2,2)
MSESS	= calMSESS(obs,pre)

##### Plot
orig_cmap	= cm.coolwarm
shifted_cmap	= shiftedColorMap(orig_cmap, midpoint=0.5, name='shifted')

plt.figure()
plt.imshow(obs.mean(0),interpolation='nearest',cmap=cm.Spectral_r)
plt.title('NLDAS-2 (%s)'%(strMon),fontsize=20)
plt.axis('off')
plt.colorbar()
plt.savefig('../Figures/NLDAS_%s.png'%(strMon), format='PNG')

plt.figure()
plt.imshow(pre.mean(0),interpolation='nearest',cmap=cm.Spectral_r)
plt.title('NMME downscaled (%s)'%(strMon),fontsize=20)
plt.axis('off')
plt.colorbar()
plt.savefig('../Figures/NMME_ds_%s.png'%(strMon), format='PNG')

plt.figure()
plt.imshow(NMME,interpolation='nearest',cmap=cm.Spectral_r)
plt.colorbar()
plt.xticks(np.arange(-0.5,1.5,0.125),[])
plt.yticks(np.arange(-0.5,1.5,0.125),[])
plt.grid(True)
plt.title('NMME (%s)'%(strMon),fontsize=20)
plt.savefig('../Figures/NMME_%s.png'%(strMon), format='PNG')

plt.figure(figsize=(10,5))
plt.plot(obs.mean(-1).mean(-1),color='LimeGreen',linewidth=1.5,marker='o',markersize=8,label='NLDAS-2')
plt.plot(pre.mean(-1).mean(-1),color='OrangeRed',linewidth=1.5,marker='o',markersize=8,label='Forecast from NMME')
plt.fill_between(np.arange(nYear),pre_Up.mean(-1).mean(-1),pre_Low.mean(-1).mean(-1),facecolor='gray',alpha=0.5,label='95% CI')
plt.plot([],[],color='gray',linewidth=10,label='95% CI',alpha=0.5)
plt.xlim([-1,nYear])
plt.xticks(range(nYear)[::5],Year[::5])
plt.xlabel('Year',fontsize=15)
plt.ylabel('mm/day',fontsize=15)
plt.title('Precipitation in %s'%(strMon),fontsize=20)
plt.legend()
plt.savefig('../Figures/PrecFore_%s.png'%(strMon), format='PNG')

plt.figure()
plt.imshow(MSESS,vmin=-0.2,vmax=0.2,cmap=shifted_cmap,interpolation='nearest')
plt.title('MSESS (%s)'%(strMon),fontsize=20)
plt.axis('off')
plt.colorbar()
plt.savefig('../Figures/MSESS_%s.png'%(strMon), format='PNG')

plt.show()

