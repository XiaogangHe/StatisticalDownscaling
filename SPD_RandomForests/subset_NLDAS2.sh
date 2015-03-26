#!/bin/sh

cat << EOF > subset.gs

'open /home/raid19/forecast/NCST/nomad6.ncep.noaa.gov/pub/raid2/wd20yx/nldas/NLDASII_Forcing/nldasforce-a-2011.ctl'
'set lat 25 35'
'set lon -89 -80'
'set t 1 last'
'set gxout fwrite'
'set fwrite cape180_0mb.bin'
'd cape180_0mb'
'disable fwrite'
'quit'

EOF
