###############################################################################
###################  Statistical Downscaling of Rainfall  #####################
###############################################################################

# load libraries
library("spTimer");

# Read data
SEUSData <- read.table("../Data/Input/SEUSData_32_syn_0.25-0.125.txt",header=TRUE)

for(i in 1982:2009){
  preYear <- c(i)
  preMon <- c(11)  # Select November as the predict month
  DataSelMon <- spT.subset(data=SEUSData, var.name=c("Month"),s=preMon)
  Data <- subset(DataSelMon, with(DataSelMon, Year==preYear))
  
  write(Data$NLDAS_prec, file = sprintf("../Data/Output/grid_32/NLDAS_obs_%s%02d.txt",preYear,preMon),ncolumns=1)
  write(Data$NLDAS_prec_UpDown, file = sprintf("../Data/Output/grid_32/NLDAS_obsUpDown_%s%02d.txt",preYear,preMon),ncolumns=1)
  
}


