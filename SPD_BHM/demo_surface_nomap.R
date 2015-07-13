################################################################
###################  New York data example #####################
################################################################

# define a pause function
pause <- function() invisible(readline("\n Please Enter to continue: \n"))

# load libraries
library("spTimer");
library("maps");
library("colorspace");

# Read data 
data(NYdata)
s<-c(8,11,12,14,18,21,24,28)
DataFit<-spT.subset(data=NYdata, var.name=c("s.index"), s=s, reverse=TRUE) 
DataFit<-subset(DataFit, with(DataFit, !(Day %in% c(30, 31) & Month == 8)))
DataValPred<-spT.subset(data=NYdata, var.name=c("s.index"), s=s) 
DataValPred<-subset(DataValPred, with(DataValPred, !(Day %in% c(30, 31) & Month == 8)))


# Figure 7
coords<-as.matrix(unique(cbind(DataFit[,2:3])))
pred.coords<-as.matrix(unique(cbind(DataValPred[,2:3])))
map(database="state",regions="new york")
points(coords,pch=19,col=3)
points(coords,pch=1,col=1)
points(pred.coords,pch=3,col=4)
legend(x=-77.5,y=41.5,col=c(3,4),pch=c(19,3),cex=0.8,legend=c("Fitted sites","Validation sites"))


# Fit GP model 
set.seed(11)
post.gp <- spT.Gibbs(formula=o8hrmax ~cMAXTMP+WDSP+RH,data=DataFit, 
        model="GP", coords=~Longitude+Latitude, scale.transform="SQRT",
		spatial.decay=spT.decay(distribution=Gamm(2,1),tuning=0.1))
print(post.gp)
# plot(post.gp)
summary(post.gp)


# Spatial prediction for the GP model
set.seed(11)
pred.gp <- predict(post.gp, newdata=DataValPred, newcoords=~Longitude+Latitude)
print(pred.gp)
names(pred.gp)

# model summary
summary(post.gp)
# validation criteria
spT.validation(DataValPred$o8hrmax,c(pred.gp$Median))  


###############################
## For surface plots         ##
## Press Enter:              ##
###############################
pause()

nItr=100
nBurn=50
# Predict on grids
data(NYgrid)
set.seed(11)
post.gp2 <- spT.Gibbs(formula=o8hrmax ~cMAXTMP+WDSP+RH,data=NYdata, 
        model="GP", coords=~Longitude+Latitude, scale.transform="SQRT",
		spatial.decay=spT.decay(distribution=Gamm(2,1),tuning=0.1))
set.seed(11)
grid.pred <- predict(post.gp2, newdata=NYgrid, newcoords=~Longitude+Latitude)

# predictive plots
library(MBA)
library(fields)
library(maps)

# this function is used to delete values outside NY
fnc.delete.map.XYZ<-function(xyz){
	x<-xyz$x; y<-xyz$y; z<-xyz$z
	xy<-expand.grid(x, y)
	eus<-(map.where(database="state", x=xy[,1], y=xy[,2]))
	dummy<-rep(0, length(xy[,1]))
	eastUS<-NULL
	eastUS<-data.frame(lon=xy[,1],lat=xy[,2],state=eus,dummy=dummy)
	eastUS[!is.na(eastUS[,3]),4]<-1
	eastUS[eastUS[,3]=="pennsylvania" & !is.na(eastUS[,3]),4]<-0
	eastUS[eastUS[,3]=="new jersey" & !is.na(eastUS[,3]),4]<-0
	eastUS[eastUS[,3]=="connecticut" & !is.na(eastUS[,3]),4]<-0
	eastUS[eastUS[,3]=="massachusetts:main" & !is.na(eastUS[,3]),4]<-0
	eastUS[eastUS[,3]=="new hampshire" & !is.na(eastUS[,3]),4]<-0
	eastUS[eastUS[,3]=="vermont" & !is.na(eastUS[,3]),4]<-0
	a <- eastUS[, 4]
	z <- as.vector(xyz$z)
	z[!a] <- NA
	z <- matrix(z, nrow = length(xyz$x))
      xyz$z <- z
      xyz
}
##

coords<-unique(NYdata[,c("Longitude","Latitude")])
grid.coords<-unique(NYgrid[,c("Longitude","Latitude")])
true.val<-matrix(NYdata$o8hrmax,62,28)
grid.val<-matrix(grid.pred$Median,62,dim(grid.coords)[[1]])
grid.sd<-matrix(grid.pred$SD,62,dim(grid.coords)[[1]])

surfplot<-function(day=60, val, ...)
{
    z <- val
	surf<-cbind(grid.coords,z[day,])
	surf<-mba.surf(surf,200,200)$xyz
	#surf<-fnc.delete.map.XYZ(xyz=surf)
	#map(database="state",regions="new york")
	image.plot(surf, xlab="Longitude",ylab="Latitude",axes=F, ...)
	contour(surf,nlevels=10,lty=3,add=T)
	map(database="state",regions="new york",add=T)
	axis(1);axis(2)
}

# prediction for day 60
day<-60
surfplot(day, val=grid.val, col = rainbow_hcl(100, start = 200, end = 0))
text(coords,labels=round(true.val[day,],1),cex=0.8,col=1)

# sd for day 60
# Press Enter:
pause()

# sd for day 60
day<-60
surfplot(day, val=grid.sd,col = diverge_hcl(100, h = c(246, 40), c = 96, l = c(65, 90)))
points(coords,pch=19,cex=1,col=2)
points(coords,pch=1,cex=1,col=1)

library(akima)
plot.spT<-function(x, residuals=FALSE, surface=NULL, time=c(1), a3d=FALSE, 
                   points=FALSE, title=TRUE, ...){
  if(is.null(surface) & a3d==FALSE){
    if(as.logical(residuals)==FALSE){
      tmp<-as.mcmc(x)
      plot(tmp, ...)}
    else{
      plot(x$fitted[,1],residuals(x),ylab="Residuals",xlab="Fitted values")
      abline(h=0,lty=2);title("Residuals vs Fitted")
      par(ask=TRUE)
      qqnorm(residuals(x));qqline(residuals(x),lty=2)}} 
  else {
    if(is.null(surface)){
      stop("\n# Error: surface should be defined as 'Mean' or 'SD'. \n")}  
    if(!surface %in% c("Mean","SD")){
      stop("\n# Error: surface only takes 'Mean' or 'SD'. \n")}
    library(akima); library(fields); 
    z<-array(fitted(x)[,paste(surface)],dim=c(x$T*x$r,x$n))
    xyz<-cbind(x$coords,c(z[time,]))
    xyz<-interp(x=xyz[,1],y=xyz[,2],z=xyz[,3],
                xo=seq(min(xyz[,1]),max(xyz[,1]),length=150),
                yo=seq(min(xyz[,2]), max(xyz[,2]), length = 150))
    if(a3d==TRUE){
      persp(x=xyz$x,y=xyz$y,z=xyz$z, xlab="x",ylab="y",zlab="z", ...)->res}
    else{
      image.plot(xyz, ...) 
      if(points != FALSE){
        points(x$coords,pch=16,cex=0.8)}}
    if(title==TRUE){
      title(main=paste("Time point: (t=",time,")",sep=""))}}}
contour.spT<-function(x, surface="Mean", time=c(1), ...){
  z<-array(fitted(x)[,paste(surface)],dim=c(x$T*x$r,x$n))
  xyz<-cbind(x$coords,c(z[time,]))
  xyz<-interp(x=xyz[,1], y=xyz[,2], z=xyz[,3], xo=seq(min(xyz[,1]), max(xyz[,1]), length = 150),
              yo=seq(min(xyz[,2]), max(xyz[,2]), length = 150),linear = TRUE, extrap=FALSE, 
              duplicate = "error", dupfun = NULL, ncp = NULL)
  contour(xyz, ...) 
}

library(sp)
data(meuse)
# model with GP
set.seed(11)
post.gp <- spT.Gibbs(formula=zinc ~ sqrt(dist),
                     data=meuse, model="GP", coords=~x+y, nItr=500, nBurn=100,
                     spatial.decay=spT.decay(distribution=Gamm(2,1), tuning=0.5),
                     distance.method="euclidean",scale.transform="LOG")
summary(post.gp)

#SEUSData <- read.table("../../Data/Input/SEUSData_16_syn.txt",header=TRUE)
#post.gp <- spT.Gibbs(formula=NLDAS_prec ~ NLDAS_prec_UpDown, 
#                     data=SEUSData, model="GP", coords=~Longitude+Latitude, 
#                     nItr=500, nBurn=100, scale.transform="SQRT", 
#                     spatial.decay=spT.decay(distribution=Gamm(2,1), tuning=0.1))

SEUSData <- read.table("../../Data/Input/SEUSData_16_syn.txt",header=TRUE)
preMon <- c(7)  # Select July as the predict month
preYear <- c(2000)
DataSelMon <- spT.subset(data=SEUSData, var.name=c("Month"),s=preMon)

DataFit <- subset(DataSelMon, with(DataSelMon, !(Year==preYear)))		# Exclude 1982 as 1982 will be the predicted year
DataValPred <- subset(DataSelMon, with(DataSelMon, Year==preYear))
post.gp <- spT.Gibbs(formula=NLDAS_prec ~ NLDAS_prec_UpDown, 
                     data=SEUSData, model="GP", coords=~Longitude+Latitude, 
                     nItr=500, nBurn=100, scale.transform="SQRT", 
                     spatial.decay=spT.decay(distribution=Gamm(2,1), tuning=0.1))

set.seed(11)


plot(post.gp, surface="Mean", title=FALSE)

