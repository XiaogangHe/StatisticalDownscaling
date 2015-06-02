#!/usr/bin/env python

import	numpy			as	np
import	matplotlib.pyplot	as	plt
from	matplotlib		import	colors
import	matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
### Plot
obs		= np.fromfile('../apcpsfc_2011_JJA_SEUS.bin', 'float32').reshape(-1, 80, 72)[:1000]

#a = np.fromfile('../prec_prediction_SEUS_RF.bin','float64').reshape(-1, 80, 72)
a = np.fromfile('../prec_prediction_SEUS_RF_adjacent_LargeMeteo_0.5deg.bin','float64').reshape(-1, 80, 72)

cpalette = np.loadtxt('../WhiteBlueGreenYellowRed.rgb',skiprows=2)/255.
cmap = colors.ListedColormap(cpalette, 256)
cmap.set_bad('0.7')

plt.figure()
plt.imshow(np.ma.masked_equal(a[-1][::-1], -9.99e+08), cmap=cmap, interpolation='nearest', vmin=0, vmax=9)
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.title('Downscaled (add adjacent prec, 0.5deg)')
plt.savefig('../../Figures/Downscaled_SEUS_lastday_adjacent_0.5deg.png', format='PNG')
#plt.savefig('../../Figures/test.pdf', format='PDF')

plt.figure()
plt.imshow(np.ma.masked_equal(obs[-1][::-1], -9.99e+08), cmap=cmap, interpolation='nearest', vmin=0, vmax=9)
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.title('Observed')
#plt.savefig('../../Figures/Observed_SEUS_lastday.png', format='PNG')
plt.show()
plt.show()
