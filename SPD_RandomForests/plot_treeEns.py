#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as	plt
import matplotlib.cbook as cbook

### Basic information
nlat_fine = 80
nlon_fine = 72
ntime = 1000
nTree = 30
sTime = 0
eTime = 100

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
