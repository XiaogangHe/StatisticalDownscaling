###############################################################################
###################  Statistical Downscaling of Rainfall  #####################
###############################################################################

# define a pause function
pause <- function() invisible(readline("\n Please Enter to continue: \n"))

# load libraries
library("spTimer");

# Read data
SEUSData <- read.table("SEUSData_16.txt",header=TRUE)
preMon <- c(8)	# Select July as the predict month
preYear <- c(2009)
DataSelMon <- spT.subset(data=SEUSData, var.name=c("Month"),s=preMon)

DataFit <- subset(DataSelMon, with(DataSelMon, !(Year==preYear)))		# Exclude 1982 as 1982 will be the predicted year
DataValPred <- subset(DataSelMon, with(DataSelMon, Year==preYear))

set.seed(11)
post.gp <- spT.Gibbs(formula=NLDAS_prec ~ NMME_prec, data=DataFit, model="GP", coords=~Longitude+Latitude, nItr=500, nBurn=100, scale.transform="SQRT", spatial.decay=spT.decay(distribution=Gamm(2,1), tuning=0.1))

#print(post.gp)
#plot(post.gp)

#autocorr.diag(as.mcmc(post.gp))
#plot(post.gp, residuals=TRUE)

# summary(post.gp)

#sink(file = sprintf("Summary_post.gp_%s%02d.txt",preYear,preMon))
#summary(post.gp) 
#sink()

set.seed(11)
pred.gp <- predict(post.gp, newdata=DataValPred, newcoords=~Longitude+Latitude, foreStep=1, type="temporal")

#print(pred.gp)
#names(pred.gp)
#spT.validation(DataValPred$NLDAS_prec, c(pred.gp$Median))
#write(DataValPred$NLDAS_prec, file = sprintf("NLDAS_obs_%s%02d.txt",preYear,preMon),ncolumns=1)
#write(pred.gp$Median, file = sprintf("NLDAS_pre_%s%02d.txt",preYear,preMon),ncolumns=1)
write(pred.gp$Mean, file = sprintf("NLDAS_pre_Mean_%s%02d.txt",preYear,preMon),ncolumns=1)
write(pred.gp$SD, file = sprintf("NLDAS_pre_SD_%s%02d.txt",preYear,preMon),ncolumns=1)
write(pred.gp$Low, file = sprintf("NLDAS_pre_Low_%s%02d.txt",preYear,preMon),ncolumns=1)
write(pred.gp$Up, file = sprintf("NLDAS_pre_Up_%s%02d.txt",preYear,preMon),ncolumns=1)
