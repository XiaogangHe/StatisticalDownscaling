'open /home/raid19/forecast/NCST/nomad6.ncep.noaa.gov/pub/raid2/wd20yx/nldas/NLDASII_Forcing/nldasforce-a-2011.ctl'

varName='apcpsfc'

'set lat 25.0625 34.9375'
'set lon -88.9375 -80.0625'
'define mask=const('%varName%',1)'
'set time 0z01jun 23z31aug'
'define up=re('%varName%',10,linear,-88.9375,1,11,linear,25.0625,1,ba)'
'define down=re(up,72,linear,-88.9375,0.125,80,linear,25.0625,0.125,ba)'
'define downMask=maskout(down,mask-0.1)'
'set gxout fwrite'
'set fwrite '%varName%'_UpDown_2011_JJA_SEUS.bin'
'd downMask'
'disable fwrite'
'quit'

