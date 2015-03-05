# define a pause function
pause <- function() invisible(readline("\n Please Enter to continue: \n"))

# load libraries
library("spTimer");
library("maps");
library("colorspace");

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

SEUSData <- read.table("~/Dropbox/Study/Princeton_2014-2015_Spring/CEE509/Data/Input/SEUSData_16_syn_0.25-0.125.txt",header=TRUE)

for(i in 1982:2009){
  preYear <- c(i)
  preMon <- c(7)  # Select July as the predict month
  DataSelMon <- spT.subset(data=SEUSData, var.name=c("Month"),s=preMon)
  
  DataFit <- subset(DataSelMon, with(DataSelMon, (Year==preYear)))	
  post.gp <- spT.Gibbs(formula=NLDAS_prec ~ NLDAS_prec_UpDown, 
                     data=DataFit, model="GP", coords=~Longitude+Latitude, 
                     nItr=500, nBurn=100, 
                     #scale.transform="LOG", 
                     spatial.decay=spT.decay(distribution=Gamm(2,1), tuning=1.5))
                     #spatial.decay=spT.decay(distribution="FIXED"))
  
  plot(post.gp)
  plot(post.gp, surface="Mean", title=FALSE)
  
  write(fitted(post.gp)$Mean, file = sprintf("../Data/Output/NLDAS_fitted_Mean_%s%02d_0.25-0.125.txt",preYear,preMon),ncolumns=1)
  write(fitted(post.gp)$SD, file = sprintf("../Data/Output/NLDAS_fitted_SD_%s%02d_0.25-0.125.txt",preYear,preMon),ncolumns=1)
  
  sink(file = sprintf("../Data/Output/Summary_fit_%s%02d_0.25-0.125.txt",preYear,preMon))
  spT.validation(DataFit$NLDAS_prec,fitted(post.gp)$Mean) 
  sink()
}
