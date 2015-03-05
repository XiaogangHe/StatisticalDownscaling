#!/usr/bin/env python

import	numpy			as	np
import	matplotlib.pyplot	as	plt
from	sklearn.metrics		import	mean_squared_error

syear	= 1982
eyear	= 2009
nyear	= eyear-syear+1
mon	= 8			# Need to modify
nLat	= 16
nLon	= 16

##### Read the data
obs	= np.array([np.loadtxt('../Data/Output/NLDAS_obs_%s%02d.txt'%(iyear,mon)).reshape(nLat,nLon) for iyear in range(syear,eyear+1)])
fit	= np.array([np.loadtxt('../Data/Output/NLDAS_fitted_Mean_%s%02d.txt'%(iyear,mon)).reshape(nLat,nLon) for iyear in range(syear,eyear+1)])

##### Analysis
rmse	= np.array([np.sqrt(mean_squared_error(obs[i],fit[i])) for i in range(nyear)])

##### Plot
for iyear in range(syear, eyear+1):
	plt.figure()
	plt.imshow(obs[iyear-syear])
	plt.title('Observed NLDAS (%s.%02d)'%(iyear,mon),fontsize=20)
	plt.colorbar()
	plt.savefig('../Figures/ObsNLDAS_%s%02d.png'%(iyear,mon),format='PNG')
	
	plt.figure()
	plt.imshow(fit[iyear-syear])
	plt.title('Fitted NLDAS (%s.%02d)'%(iyear,mon),fontsize=20)
	plt.colorbar()
	plt.savefig('../Figures/FitNLDAS_%s%02d.png'%(iyear,mon),format='PNG')

#plt.plot(obs.mean(-1).mean(-1))
#plt.plot(fit.mean(-1).mean(-1))

plt.figure()
plt.plot(rmse)
plt.xlim([-1,nyear])
plt.xticks(range(nyear)[::5],np.arange(syear,eyear+1)[::5])
plt.ylabel('RMSE (mm/day)')
plt.title('%02d'%(mon),fontsize=20)
plt.savefig('../Figures/RMSE_%02d.png'%(mon),format='PNG')

plt.show()
