'open /home/raid19/forecast/NCST/nomad6.ncep.noaa.gov/pub/raid2/wd20yx/nldas/NLDASII_Forcing/nldasforce-a-2011.ctl'

* variable Name list = ['apcpsfc', 'cape180_0mb', 'pressfc', 'spfh2m', 'tmp2m', 'ugrd10m', 'vgrd10m']
varName='cape180_0mb'
res='1deg'

'set lat 25.0625 35.0625'
'set lon -88.9375 -80.0625'
'define mask=const('%varName%',1)'
'set time 0z01jun 23z31aug'
'define up=re('%varName%',10,linear,-88.9375,1,12,linear,25.0625,1,ba)'
'define down=re(up,73,linear,-88.9375,0.125,81,linear,25.0625,0.125,ba)'
'define up=re(down,10,linear,-88.9375,1,12,linear,25.0625,1,ba)'
'define down=re(up,73,linear,-88.9375,0.125,81,linear,25.0625,0.125,bl)'
'define downMask=maskout(down,mask-0.1)'
'set lat 25.0625 34.9375'
'set gxout fwrite'
'set fwrite '%varName%'_UpDown_'%res%'_2011_JJA_SEUS_bi-linear.bin'
'd downMask'
'disable fwrite'
'quit'

