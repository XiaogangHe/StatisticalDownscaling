#!/usr/bin/env python

import	numpy				as	np
import	matplotlib.pyplot		as	plt
import	time
import	os
from	pandas				import	DataFrame
from	sklearn.ensemble		import	RandomForestClassifier
from	sklearn.metrics			import	classification_report
from 	sklearn.metrics			import	recall_score
from	sklearn.metrics			import	roc_curve, auc

### Read datasets
slope = np.fromfile('/home/wind/hexg/Research/Data/NLDAS2/slope_SEUS.bin','float32').reshape(82,74)
aspect = np.fromfile('/home/wind/hexg/Research/Data/NLDAS2/aspect_SEUS.bin','float32').reshape(82,74)
gtopomean = np.fromfile('/home/wind/hexg/Research/Data/NLDAS2/gtopomean_SEUS.bin','float32').reshape(82,74)
gtopostd = np.fromfile('/home/wind/hexg/Research/Data/NLDAS2/gtopostd_SEUS.bin','float32').reshape(82,74)

obs_fine = np.loadtxt('/home/wind/hexg/Research/Data/NLDAS2/NLDAS_obs_199206.txt')
obs_coarse = np.loadtxt('/home/wind/hexg/Research/Data/NLDAS2/NLDAS_obsUpDown_199206.txt')
obs_coarse_pre = np.loadtxt('/home/wind/hexg/Research/Data/NLDAS2/NLDAS_obsUpDown_199306.txt')

features_train = DataFrame({
		'prec_UpDown':obs_coarse,
		'slope': slope[::-1][:16,:16].reshape(-1),
		'aspect':aspect[::-1][:16,:16].reshape(-1),
		'gtopomean':gtopomean[::-1][:16,:16].reshape(-1),
		'gtopostd':gtopostd[::-1][:16,:16].reshape(-1)
		})
features_test = DataFrame({
		'prec_UpDown':obs_coarse_pre,
		'slope': slope[::-1][:16,:16].reshape(-1),
		'aspect':aspect[::-1][:16,:16].reshape(-1),
		'gtopomean':gtopomean[::-1][:16,:16].reshape(-1),
		'gtopostd':gtopostd[::-1][:16,:16].reshape(-1)
		})

### Random forests
clf_RF		= RandomForestClassifier(n_estimators=200,random_state=0)

time_start      = time.time()
clfRFFit	= clf_RF.fit(features_train, obs_fine)
time_tr         = time.time()
pre_te_RF	= clf_RF.predict(features_test)
time_te         = time.time()

plt.figure()
plt.imshow(pre_te_RF.reshape(16,16))
plt.colorbar()
plt.title('Prediction (RF)')
plt.show()
