#!/usr/bin/env python

import	numpy               as	np
import	matplotlib.pyplot   as	plt
import  pandas.rpy.common   as  com
from    rpy2.robjects       import r
from    rpy2.robjects.packages  import importr

##### Load R packages
importr('ncf')

x = r("x <- expand.grid(1:20, 1:5)[,1]")
y = r("y <- expand.grid(1:20, 1:5)[,2]")
z = r("z <- rmvn.spa(x=x, y=y, p=2, method='exp')")
fit = r("spline.correlog(x=x, y=y, z=z, resamp=100)")
fit = com.convert_robj(fit)

plt.figure()
plt.plot(fit['real']['predicted']['x'], fit['real']['predicted']['y'])
plt.show()
