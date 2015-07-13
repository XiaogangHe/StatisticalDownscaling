library(spTimer)

# These packages will be required to run the code in this file
library(akima)
library(coda)
library(spacetime)
library(fields)
library(forecast) 
library(MASS)
library(mgcv)
library(spBayes)


# define a pause function
pause <- function() invisible(readline("\n Please Enter to continue: \n"))

#######################################################################################
################################## Simulation study  ##################################
#######################################################################################

pause()

############################ Functions to simulate data  ##############################
#######################################################################################

## Function to simulate data: GP model

data.sim.GP<-function(n, r=1, T, sig2e=0.01, sig2eta=0.1, phi=0.003, beta=5.0){
   #set.seed(11)
   library("spTimer")
   n <- n*n # say, sites
   Longitude<-seq(0,1000,by=1000/(sqrt(n)-1))
   Latitude<-seq(0,1000,by=1000/(sqrt(n)-1))
   long.lat<-expand.grid(Longitude,Latitude)
   site<-data.frame(s.index=1:n,Longitude=long.lat[,1],Latitude=long.lat[,2])
   d<-as.matrix(dist(site[,2:3], method="euclidean", diag = TRUE, upper = TRUE))
   library(MASS)
   r <- r # year
   T<- T # day
   N<-n*r*T
   sig2e<-sig2e; sig2eta<-sig2eta; phi<-phi; D1<-exp(-phi*d); beta<-beta
   Ivec<-rep(1,n); z<-matrix(NA,r*T,n); o<-matrix(NA,r*T,n)
   if(length(beta)>1){
     x<-rep(1,N) 
     for(i in 2:length(beta)){
      x <- cbind(x,rnorm(N))
	 }
	 xb<-x%*%matrix(beta)
	 xb<-matrix(c(xb),r*T,n)
     for(i in 1:(r*T)){
       o[i,]<-xb[i,]+mvrnorm(1,rep(0,n),sig2eta*D1)
       z[i,]<-o[i,]+rnorm(1,0,sqrt(sig2e)) 
     }
     dat1<-matrix(NA,n*r*T,4+length(beta)-1)
	 dat1[,5:(4+length(beta)-1)]<-x[,-1]
     dimnames(dat1)[[2]][5:(4+length(beta)-1)]<-paste("x",1:(length(beta)-1),sep="")
   }
   else{
     for(i in 1:(r*T)){
       o[i,]<-beta+mvrnorm(1,rep(0,n),sig2eta*D1)
       z[i,]<-o[i,]+rnorm(1,0,sqrt(sig2e)) 
     }
     dat1<-matrix(NA,n*r*T,4)
   }
   dat1[,1]<-sort(rep(1:n,r*T))
   dat1[,2]<-sort(rep(1:r,T))
   dat1[,3]<-1:T
   dat1[,4]<-c(z)
   dimnames(dat1)[[2]][1:4]<-c("s.index","year","day","y")
   dat1<-as.data.frame(dat1)
   dat1<-merge(dat1,site,by=c("s.index"),all.x=TRUE)
   dat1$y_no_mis<-dat1$y
   #set.seed(11)
   dat1[sample(1:dim(dat1)[[1]],round(dim(dat1)[[1]]*0.05)),4]<-NA # 5% missing values to put
   dat1<-dat1[order(dat1$s.index,dat1$year,dat1$day),]
   dat1
}

## Function to simulate data: AR model

data.sim.AR<-function(n, r=1, T, sig2e=0.01, sig2eta=0.1, phi=0.003, beta=5.0, rho=0.2, mu=5.0,sig2=0.5){
   #set.seed(111)
   library("spTimer")
   n <- n*n # say, sites
   Longitude<-seq(0,1000,by=1000/(sqrt(n)-1))
   Latitude<-seq(0,1000,by=1000/(sqrt(n)-1))
   long.lat<-expand.grid(Longitude,Latitude)
   site<-data.frame(s.index=1:n,Longitude=long.lat[,1],Latitude=long.lat[,2])
   d<-as.matrix(dist(site[,2:3], method="euclidean", diag = TRUE, upper = TRUE))
   library(MASS)
   r <- r # year
   T<- T # day
   N<-n*r*T
   sig2e<-sig2e; sig2eta<-sig2eta; phi<-phi; D1<-exp(-phi*d); beta<-beta; rho<-rho; mu<-mu; sig2<-sig2;
   z<-matrix(NA,T*r,n); o<-matrix(NA,(T+1)*r,n); 
   if(length(beta)>1){
     x<-rep(1,N) 
     for(i in 2:length(beta)){
      x <- cbind(x,rnorm(N))
	 }
	 xb<-x%*%matrix(beta)
	 xb<-matrix(c(xb),r*T,n)
     for(j in 1:r){
      o[1+(j-1)*T,] <- mvrnorm(1,rep(mu,n),sig2*D1)
      for(i in 1:T){
       o[(i+1)+(j-1)*T,]<-rho*o[i+(j-1)*T,]+xb[i+(j-1)*T,]+mvrnorm(1,rep(0,n),sig2eta*D1)
       z[i+(j-1)*T,]<-o[(i+1)+(j-1)*T,]+rnorm(1,0,sqrt(sig2e)) 
      } 
     }
     dat1<-matrix(NA,n*r*T,4+length(beta)-1)
	 dat1[,5:(4+length(beta)-1)]<-x[,-1]
     dimnames(dat1)[[2]][5:(4+length(beta)-1)]<-paste("x",1:(length(beta)-1),sep="")
   }
   else{
     Ivec<-rep(1,n); 
     for(j in 1:r){
      o[1+(j-1)*T,] <- mvrnorm(1,rep(mu,n),sig2*D1)
      for(i in 1:T){
       o[(i+1)+(j-1)*T,]<-rho*o[i+(j-1)*T,]+beta*Ivec+mvrnorm(1,rep(0,n),sig2eta*D1)
       z[i+(j-1)*T,]<-o[(i+1)+(j-1)*T,]+rnorm(1,0,sqrt(sig2e)) 
      } 
     }
     dat1<-matrix(NA,n*r*T,4)
   }
   dat1[,1]<-sort(rep(1:n,r*T))
   dat1[,2]<-sort(rep(1:r,T))
   dat1[,3]<-1:T
   dat1[,4]<-c(z)
   dimnames(dat1)[[2]][1:4]<-c("s.index","year","day","y")
   dat1<-as.data.frame(dat1)
   dat1<-merge(dat1,site,by=c("s.index"),all.x=TRUE)
   dat1$y_no_mis<-dat1$y
   #set.seed(111)
   dat1[sample(1:dim(dat1)[[1]],round(dim(dat1)[[1]]*0.05)),4]<-NA # 5% missing values to put
   dat1<-dat1[order(dat1$s.index,dat1$year,dat1$day),]
   dat1
}

## Function to simulate data: GPP based model

data.sim.GPP<-function(n, m=10, r=1, T, sig2e=0.01, sig2eta=0.1, phi=0.003, beta=5.0, rho=0.2, mu=5.0,sig2=0.5){
   #set.seed(33)
   library("spTimer"); library("MASS")
   n <- n*n # say, sites
   Longitude<-seq(0,1000,by=1000/(sqrt(n)-1))
   Latitude<-seq(0,1000,by=1000/(sqrt(n)-1))
   long.lat<-expand.grid(Longitude,Latitude)
   knots.coords<-spT.grid.coords(Longitude=c(990.75,9.25),Latitude=c(990.75,9.25),by=c(m, m))
   spT.check.locations(fit.locations=as.matrix(long.lat),pred.locations=knots.coords,method="euclidean",tol=0.05)
   site<-data.frame(s.index=1:n,Longitude=long.lat[,1],Latitude=long.lat[,2])
   d<-as.matrix(dist(site[,2:3], method="euclidean", diag = TRUE, upper = TRUE))
   d2<-as.matrix(dist(knots.coords, method="euclidean", diag = TRUE, upper = TRUE))
   r <- r # year
   T<- T # day
   N<-n*r*T
   sig2e<-sig2e; sig2eta<-sig2eta; phi<-phi; D1<-exp(-phi*d); beta<-beta; rho<-rho; mu<-mu; sig2<-sig2;
   D2<-exp(-phi*d2); m<-m*m; 
   dd<-as.matrix(dist(rbind(as.matrix(site[,2:3]),knots.coords),method="euclidean", diag = TRUE, upper = TRUE))
   C<-dd[1:dim(site)[[1]],(dim(site)[[1]]+1):(dim(site)[[1]]+dim(knots.coords)[[1]])]
   A<-exp(-phi*C)%*%solve(D2)
   z<-matrix(NA,T*r,n); w<-matrix(NA,(T+1)*r,m); 
   if(length(beta)>1){
     x<-rep(1,N) 
     for(i in 2:length(beta)){
      x <- cbind(x,rnorm(N))
	 }
	 xb<-x%*%matrix(beta)	
	 xb<-matrix(c(xb),r*T,n)
     for(j in 1:r){
       w[1+(j-1)*T,] <- mvrnorm(1,rep(0,m),sig2*D2)
       for(i in 1:T){
         w[(i+1)+(j-1)*T,]<-rho*w[i+(j-1)*T,]+mvrnorm(1,rep(0,m),sig2eta*D2)
       } 
     }
     for(j in 1:r){
       for(i in 1:T){
         e<-rnorm(1,0,sqrt(sig2e))
         z[i+(j-1)*T,]<-A%*%w[(i+1)+(j-1)*T,]+xb[i+(j-1)*T,]+e 
       }
     }
     dat1<-matrix(NA,n*r*T,4+length(beta)-1)
	 dat1[,5:(4+length(beta)-1)]<-x[,-1]
     dimnames(dat1)[[2]][5:(4+length(beta)-1)]<-paste("x",1:(length(beta)-1),sep="")
   }
   else{
     Ivec<-rep(1,n); 
     for(j in 1:r){
       w[1+(j-1)*T,] <- mvrnorm(1,rep(0,m),sig2*D2)
       for(i in 1:T){
         w[(i+1)+(j-1)*T,]<-rho*w[i+(j-1)*T,]+mvrnorm(1,rep(0,m),sig2eta*D2)
       } 
     }
     for(j in 1:r){
       for(i in 1:T){
         e<-rnorm(1,0,sqrt(sig2e))
         z[i+(j-1)*T,]<-A%*%w[(i+1)+(j-1)*T,]+beta*Ivec+e 
       }
     }
     dat1<-matrix(NA,n*r*T,4)
   }
   dat1[,1]<-sort(rep(1:n,r*T))
   dat1[,2]<-sort(rep(1:r,T))
   dat1[,3]<-1:T
   dat1[,4]<-c(z)
   dimnames(dat1)[[2]][1:4]<-c("s.index","year","day","y")
   dat1<-as.data.frame(dat1)
   dat1<-merge(dat1,site,by=c("s.index"),all.x=TRUE)
   dat1$y_no_mis<-dat1$y
   #set.seed(33)
   dat1[sample(1:dim(dat1)[[1]],round(dim(dat1)[[1]]*0.05)),4]<-NA # 5% missing values to put
   dat1<-dat1[order(dat1$s.index,dat1$year,dat1$day),]
   dat1
}


############################### Figure 1(a) and 1(b)  #################################
#######################################################################################

pause()

## spatial domain: Figure 1(a)
pause()

library("spTimer")
n <- 12*12 # say, sites
Longitude<-seq(0,1000,by=1000/(12-1))
Latitude<-seq(0,1000,by=1000/(12-1))
long.lat<-expand.grid(Longitude,Latitude)
plot(long.lat,xlab="Longitude",ylab="Latitude",pch=3,col=4)

## spatial domain: Figure 1(b)
pause()

library("spTimer")
n <- 55*55 # say, sites
Longitude<-seq(0,1000,by=1000/(55-1))
Latitude<-seq(0,1000,by=1000/(55-1))
long.lat<-expand.grid(Longitude,Latitude)
knots.coords<-spT.grid.coords(Longitude=c(990,10),Latitude=c(990,10),by=c(10,10))
spT.check.locations(fit.locations=as.matrix(long.lat),pred.locations=knots.coords,method="euclidean",tol=1.5)
plot(long.lat,xlab="Longitude",ylab="Latitude",pch=3,col=4,cex=0.6)
points(knots.coords,pch=19,col=2)


################################### Replications ######################################
#######################################################################################

# To obtain results quickly I have changed the following numbers

replic <-2 # number of replications of the datasets, used as 25 in the paper
nItr <- 200 # number of MCMC samples for each model, used as 5000 in the paper
nBurn <- 100 # number of burn-in from the MCMC samples, used as 1000 in the paper 

# To 
pause()

## For GP model

paraGP<-NULL
for(i in 1:replic){
   set.seed(round(rnorm(1,mean=i,sd=100)))
   dat<-data.sim.GP(n=12,T=365,beta=c(5.0, 2.0, 1.0, 0.5),sig2eta=runif(1,0,1))
   out <- spT.Gibbs(formula=y~x1+x2+x3,data=dat,model="GP",coords=~Longitude+Latitude,distance.method="euclidean",nItr=nItr,nBurn=nBurn,report=1,spatial.decay=spT.decay(distribution=Gamm(2,1),tuning=0.9))
   paraGP<-rbind(paraGP,as.mcmc(out)[,1:(dim(as.mcmc(out))[[2]]-3)])
}

## For AR model

paraAR<-NULL
for(i in 1:replic){
   set.seed(round(rnorm(1,mean=i,sd=100)))
   dat<-data.sim.AR(n=12,T=365,beta=c(5.0, 2.0, 1.0, 0.5),sig2eta=runif(1,0,1))
   out <- spT.Gibbs(formula=y~x1+x2+x3,data=dat,model="AR",coords=~Longitude+Latitude,distance.method="euclidean",nItr=nItr,nBurn=nBurn,report=1,spatial.decay=spT.decay(distribution=Gamm(2,1),tuning=0.2))
   paraAR<-rbind(paraAR,as.mcmc(out)[,1:(dim(as.mcmc(out))[[2]]-3)])
}

## For GPP model

paraGPP<-NULL
for(i in 1:replic){
   set.seed(round(rnorm(1,mean=i,sd=100)))
   dat<-data.sim.GPP(n=55,T=365,beta=c(5.0, 2.0, 1.0, 0.5),sig2eta=runif(1,0,1))
   knots.coords<-spT.grid.coords(Longitude=c(990.75,9.25),Latitude=c(990.75,9.25),by=c(10,10))
   out <- spT.Gibbs(formula=y~x1+x2+x3,data=dat,model="GPP",coords=~Longitude+Latitude,knots.coords=knots.coords,distance.method="euclidean",nItr=nItr,nBurn=nBurn,report=1,tol.dist=1,spatial.decay=spT.decay(distribution=Gamm(2,1),tuning=0.08))
   paraGPP<-rbind(paraGPP,as.mcmc(out)[,1:(dim(as.mcmc(out))[[2]]-3)])
}

##################################### Table 2  ########################################
#######################################################################################

pause()

## summary of parameter estimates: Table 2

t(apply(paraGP,2,quantile,prob=c(0.025,0.5,0.975)))
t(apply(paraAR,2,quantile,prob=c(0.025,0.5,0.975)))
t(apply(paraGPP,2,quantile,prob=c(0.025,0.5,0.975)))

############################### Figure 2(a) and 2(b)  #################################
#######################################################################################

pause()

## Surface plot: Figure 2(a) and 2(b)

set.seed(round(rnorm(1,mean=i,sd=100)))
n<-18; T<-30
dat<-data.sim.GPP(n=n,T=T,beta=c(5.0, 2.0, 1.0, 0.5))
knots.coords<-spT.grid.coords(Longitude=c(990.75,9.25),Latitude=c(990.75,9.25),by=c(7,7))
out <- spT.Gibbs(formula=y~x1+x2+x3,data=dat,model="GPP",coords=~Longitude+Latitude,knots.coords=knots.coords,distance.method="euclidean",nItr=nItr,nBurn=nBurn,report=1,spatial.decay=spT.decay(distribution=Gamm(2,1),tuning=0.08))

# S3 class code for plot.spT and contour.spT using R package akima
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
  
# For Mean
library(colorspace)
time <- 5
plot(out, surface = "Mean", time = time, col=terrain_hcl(12), legend.shrink = 0.5, legend.width = 0.8,horizontal=TRUE, title=FALSE)
#contour(out,add=TRUE,lty=2)

# For surface error plot
err <- array(dat$y_no_mis-out$fitted[,1], dim = c(T, n*n))
xyz <- cbind(unique(dat[,c("Longitude","Latitude")]), c(err[time, ]))
xyz <- interp(x = xyz[, 1], y = xyz[, 2], z = xyz[, 3], 
       xo = seq(min(xyz[, 1]), max(xyz[, 1]), length = 150), 
       yo = seq(min(xyz[, 2]), max(xyz[, 2]), length = 150), 
       linear = TRUE, extrap = FALSE, duplicate = "error", 
       dupfun = NULL, ncp = NULL)
image.plot(xyz, col=diverge_hcl(7, h = c(246, 40), c = 96, l = c(65, 90)), legend.shrink = 0.5, legend.width = 0.8,horizontal=TRUE)


############################### Figure 3(a) and 3(b)  #################################
#######################################################################################

pause()

## Time-series plot: Figure 3(a) 

pause()

z1 <- array(dat$y, dim = c(T, n*n))
z2 <- array(dat$y_no_mis, dim = c(T, n*n))
z3 <- array(c(out$fitted[,1]), dim = c(T, n*n))
s.index<-1
zz<-cbind(1:length(c(z3[,s.index])),c(z3[,s.index]),c(z1[,s.index]),c(z2[,s.index]))
plot(zz[,4],type="o",xlab="Time series", pch=16,ylab="y",axes=FALSE,cex=0.9,ylim=c(0,10),lty=2,lwd=1.5)
points(zz[,2],pch=12,col=2,type="o",lty=1,lwd=1)
# Please note that if 
zzz<-zz[is.na(zz[,3]),]
if(is.null(dim(zzz))){
  points(zzz[1],zzz[4],pch=1,cex=2,col=4)
}
if(!is.null(dim(zzz))){
  points(zzz[,1],zzz[,4],pch=1,cex=2,col=4)
}
axis(2);axis(1,1:T,labels=1:T)
legend("topleft",pch=c(16,12,1),col=c(1,2,4),lty=c(2,1,NA),bty="n",cex=1,
       legend=c("True values","Fitted values","Missing values"))

## Time-series plot: Figure 3(b)

pause()


plot(zz[,2]-zz[,4],type="n",xlab="Time series", pch=16,ylab="Error",axes=FALSE,cex=0.7,ylim=c(-0.8,0.8),lty=1,lwd=0.5)
s.indexx<-c(1,2,3,6) # chosen 4 locations 1,2,3,6
for(s.index in s.indexx){
zz<-cbind(1:length(c(z3[,s.index])),c(z3[,s.index]),c(z1[,s.index]),c(z2[,s.index]))
lines(zz[,2]-zz[,4],type="o",xlab="Time series", pch=s.index+1,ylab="y",cex=0.7,lty=s.index,col=s.index)
zzz<-zz[is.na(zz[,3]),]
if(is.null(dim(zzz))){
  points(zzz[1],zzz[2]-zzz[4],pch=1,cex=2,col=4)
}
if(!is.null(dim(zzz))){
  points(zzz[,1],zzz[,2]-zzz[,4],pch=1,cex=2,col=4)
}
}
legend("topleft",pch=c(2,3,4,7,1),lty=c(2,3,4,6,NA),col=c(1,2,3,6,4),bty="n",cex=1,
       legend=c("Series 1","Series 2","Series 3","Series 4","Missing"))
axis(2);axis(1,1:T,labels=1:T)


	   
##################################### Figure 4  #######################################
#######################################################################################

pause()

## Sensitivity analysis for signal-to-noise ratio (SNR)

# SNR = sig2eta/sig2e = 0.01/0.01 = 1
ptm <- proc.time()
para1<-NULL
stn1<-NULL
for(i in 1:replic){
	dat<-data.sim.GP(n=12,T=365,beta=c(5.0),sig2eta=0.01)
    s<-sample(1:(12*12),15)
    fit<-spT.subset(data=dat, var.name="s.index", s = s, reverse = TRUE) # for model fitting
	# model fitting with spTimer
	library(spTimer)
	out<-spT.Gibbs(formula=y~1,data=fit,coords=~Longitude+Latitude,nItr=nItr,nBurn=nBurn,report=1,distance.method="euclidean",spatial.decay=spT.decay(distribution=Gamm(2,1),tuning=0.2))
	para1<-rbind(para1,as.mcmc(out)[,1])
	stn1<-c(stn1,as.mcmc(out)[,3]/as.mcmc(out)[,2])
}
paraGP1<-quantile(c(para1), probs=c(0.025,0.5,0.975))
rm(out)
proc.time() - ptm

# SNR = sig2eta/sig2e = 0.1/0.01 = 10
ptm <- proc.time()
para10<-NULL
stn10<-NULL
for(i in 1:replic){
	dat<-data.sim.GP(n=12,T=365,beta=c(5.0),sig2eta=0.1)
    s<-sample(1:(12*12),15)
    fit<-spT.subset(data=dat, var.name="s.index", s = s, reverse = TRUE) # for model fitting
	# model fitting with spTimer
	library(spTimer)
	out<-spT.Gibbs(formula=y~1,data=fit,coords=~Longitude+Latitude,nItr=nItr,nBurn=nBurn,report=1,distance.method="euclidean",spatial.decay=spT.decay(distribution=Gamm(2,1),tuning=0.1))
	para10<-rbind(para10,as.mcmc(out)[,1])
	stn10<-c(stn10,as.mcmc(out)[,3]/as.mcmc(out)[,2])
}
paraGP10<-quantile(c(para10), probs=c(0.025,0.5,0.975))
rm(out)
proc.time() - ptm

# SNR = sig2eta/sig2e = 0.15/0.01 = 15

ptm <- proc.time()
para15<-NULL
stn15<-NULL
for(i in 1:replic){
	dat<-data.sim.GP(n=12,T=365,beta=c(5.0),sig2eta=0.25)
    s<-sample(1:(12*12),15)
    fit<-spT.subset(data=dat, var.name="s.index", s = s, reverse = TRUE) # for model fitting
	# model fitting with spTimer
	library(spTimer)
	out<-spT.Gibbs(formula=y~1,data=fit,coords=~Longitude+Latitude,nItr=nItr,nBurn=nBurn,report=1,distance.method="euclidean",spatial.decay=spT.decay(distribution=Gamm(2,1),tuning=0.1))
	para15<-rbind(para15,as.mcmc(out)[,1])
	stn15<-c(stn15,as.mcmc(out)[,3]/as.mcmc(out)[,2])
}
paraGP15<-quantile(c(para15), probs=c(0.025,0.5,0.975))
rm(out)
proc.time() - ptm

## SNR density plot: Code for Figure 4

pause()

plot(density(c(stn1)),type="n",xlim=c(0,25),main="Signal-to-noise ratio (SNR)")
lines(density(c(stn1)),lty=2,col=2,lwd=2); abline(v=1,lty=3)
lines(density(c(stn10)),lty=4,col=4,lwd=2); abline(v=10,lty=3)
lines(density(c(stn15)),lty=6,col=6,lwd=2); abline(v=15,lty=3)
legend("topright",lty=c(2,4,6,3),col=c(2,4,6,1),bty="n",cex=0.8,lwd=1.5,
		legend=c("Distribution of SNR for 1", "Distribution of SNR for 10", 
		"Distribution of SNR for 15", "True SNR values"))
text(x=1,y=0.5,label="SNR = 1",cex=0.8)
text(x=10,y=0.5,label="SNR = 10",cex=0.8)
text(x=15,y=0.5,label="SNR = 15",cex=0.8)


##################################### Table 3  ########################################
#######################################################################################

pause()

## SNR: Table 3

paraGP1
paraGP10
paraGP15

#######################################################################################
################################## Comparison study  ##################################
#######################################################################################

pause()

## Function to simulate data GP model without grid points

data.sim.GP.nogrid<-function(nn, r=1, T, sig2e=0.01, sig2eta=0.1, phi=0.003, beta=5.0){
   #set.seed(11)
   library("spTimer")
   n <- nn # say, sites
   Longitude<-sample(0:1000,n)
   Latitude<-sample(0:1000,n)
   long.lat<-cbind(Longitude,Latitude)
   site<-data.frame(s.index=1:n,Longitude=long.lat[,1],Latitude=long.lat[,2])
   d<-as.matrix(dist(site[,2:3], method="euclidean", diag = TRUE, upper = TRUE))
   library(MASS)
   r <- r # year
   T<- T # day
   N<-n*r*T
   sig2e<-sig2e; sig2eta<-sig2eta; phi<-phi; D1<-exp(-phi*d); beta<-beta
   Ivec<-rep(1,n); z<-matrix(NA,r*T,n); o<-matrix(NA,r*T,n)
   if(length(beta)>1){
     x<-rep(1,N) 
     for(i in 2:length(beta)){
      x <- cbind(x,rnorm(N))
	 }
	 xb<-x%*%matrix(beta)
	 xb<-matrix(c(xb),r*T,n)
     for(i in 1:(r*T)){
       o[i,]<-xb[i,]+mvrnorm(1,rep(0,n),sig2eta*D1)
       z[i,]<-o[i,]+rnorm(1,0,sqrt(sig2e)) 
     }
     dat1<-matrix(NA,n*r*T,4+length(beta)-1)
	 dat1[,5:(4+length(beta)-1)]<-x[,-1]
     dimnames(dat1)[[2]][5:(4+length(beta)-1)]<-paste("x",1:(length(beta)-1),sep="")
   }
   else{
     for(i in 1:(r*T)){
       o[i,]<-beta+mvrnorm(1,rep(0,n),sig2eta*D1)
       z[i,]<-o[i,]+rnorm(1,0,sqrt(sig2e)) 
     }
     dat1<-matrix(NA,n*r*T,4)
   }
   dat1[,1]<-sort(rep(1:n,r*T))
   dat1[,2]<-sort(rep(1:r,T))
   dat1[,3]<-1:T
   dat1[,4]<-c(z)
   dimnames(dat1)[[2]][1:4]<-c("s.index","year","day","y")
   dat1<-as.data.frame(dat1)
   dat1<-merge(dat1,site,by=c("s.index"),all.x=TRUE)
   dat1$y_no_mis<-dat1$y
   #set.seed(11)
   dat1[sample(1:dim(dat1)[[1]],round(dim(dat1)[[1]]*0.05)),4]<-NA # 5% missing values to put
   dat1<-dat1[order(dat1$s.index,dat1$year,dat1$day),]
   dat1
}

#######################################################################################

library("spTimer")
library("spBayes")
library("mgcv")

## creating spBayes fnc to run the programme in the paper

run.spBayes<-function(nItr,data){
   fit<-data
   library(spBayes)
   coords<-unique(cbind(fit$Longitude,fit$Latitude))
   # spBayes cannot handle the missing values in y automatically for multivariate case
   y<-matrix(fit$y,time,dim(fit)[[1]]/time)
   y<-cbind(c(y),rep(apply(y,1,mean,na.rm=T),dim(fit)[[1]]/time))
   y[is.na(y[, 1]), 1] <- y[is.na(y[, 1]), 2]
   y[is.na(y[, 1]), 1] <- median(y[, 2], na.rm = TRUE)
   y<-matrix(y[,1],time,dim(fit)[[1]]/time)
   x1<-matrix(1,time,dim(fit)[[1]]/time)
   f<-NULL
   # need to supply equation for each day
   for(i in 1:time){
      f[[i]]<- as.formula(paste("y[",i,",]~x1[",i,",]-1",sep=""))
   }
   if(time>1){
   # Call spMvLM for more than one time points
   q<-time
   A.starting <- diag(1,q)[lower.tri(diag(1,q), TRUE)]
   n.samples <- nItr
   starting <- list("phi"=rep(3/0.5,q), "A"=A.starting, "Psi"=rep(1,q))
   tuning <- list("phi"=rep(50,q), "A"=rep(0.0001,length(A.starting)), "Psi"=rep(50,q))
   priors <- list("beta.Flat", "phi.Unif"=list(rep(3/0.75,q), rep(3/0.25,q)),
               "K.IW"=list(q+1, diag(0.1,q)), "Psi.ig"=list(rep(2,q), rep(0.1,q)))
   out2 <- spMvLM(f, 
     coords=coords, starting=starting, tuning=tuning, priors=priors,
     n.samples=nItr, cov.model="exponential", n.report=nItr)
   out2
   }
   else{
   # Call spLM for univariate model
   starting <- list("phi"=3/0.5, "sigma.sq"=50, "tau.sq"=1)
   tuning <- list("phi"=0.1, "sigma.sq"=0.1, "tau.sq"=0.1)
   priors <- list("beta.Flat", "phi.Unif"=c(3/1, 3/0.1),
               "sigma.sq.IG"=c(2, 5), "tau.sq.IG"=c(2, 0.01))
   out2 <- spLM(f[[1]], 
     coords=coords, starting=starting, tuning=tuning, priors=priors,
     n.samples=nItr, cov.model="exponential", n.report=nItr)
   out2
   }
}


############################### Figures 5(a) and 5(b)  ################################
#######################################################################################

pause()

## create data set in Grids: Figure 5(a)
pause()

n<-7
nn <- n*n # say, sites
Longitude<-seq(0,1000,by=1000/(sqrt(nn)-1))
Latitude<-seq(0,1000,by=1000/(sqrt(nn)-1))
long.lat<-expand.grid(Longitude,Latitude)
plot(long.lat,xlab="Longitude",ylab="Latitude",pch=3,col=4)

## create data set randomly without Grids: Figure 5(b)
pause()

dat<-data.sim.GP.nogrid(nn=n*n,T=time,beta=c(5.0))
plot(unique(dat[,c("Longitude","Latitude")]),xlab="Longitude",ylab="Latitude",pch=3,col=4)
rm(dat)

#######################################################################################

## Comparison study

pause()

n<-7
time<-20 # takes 5, 10, 20, 60
dat<-data.sim.GP(n=n,T=time,beta=c(5.0))

## spTimer
pause()

out1<-spT.Gibbs(formula=y~1,data=dat,coords=~Longitude+Latitude,nItr=nItr,nBurn=nBurn,distance.method="euclidean",spatial.decay=spT.decay(distribution=Gamm(2,1),tuning=0.1))

## spBayes
pause()

start.time <- proc.time()[3]
out2<-run.spBayes(nItr=nItr,data=dat)
out2<- spRecover(out2, start=nBurn, verbose=FALSE)
end.time <- proc.time()[3]
comp.time <- end.time - start.time
comp.time
 
## GAM using mgcv
pause()

out3<-gam(formula=y~s(Longitude,Latitude),data=dat)


############################### Figure 6(a) and 6(b)  ################################
#######################################################################################

pause()

## density plots for time=5 and time=10
## change time points as 5 and 10

tmp<-out2$p.beta.recover.samples
plot(density(tmp),col=4,lty=4,main=paste("Data with time point = ",time,sep=""),xlab="",ylim=c(0,6),xlim=c(0,10))
lines(density(c(out1$betap)),col=2,lty=2); 
abline(v=coef(out3)[[1]],lty=3)
abline(v=5,lty=1)
legend("topleft",col=c(1,1,4,2),lty=c(1,3,4,2),bty="n",cex=0.8,
      legend=c("True","GAM","spBayes","spTimer"))

#######################################################################################

## Predictions validations 
## Grid data

pause()

## Grid data

n<-7
time<-5 # takes 5, and 60

#set.seed(22)
dat<-data.sim.GP(n=n,T=time,beta=c(5.0))

# Model fitting
#set.seed(11)
s<-sample(1:(n*n),n*n*0.2)
val<-spT.subset(data=dat, var.name="s.index", s = s, reverse = FALSE) # for model validation
fit<-spT.subset(data=dat, var.name="s.index", s = s, reverse = TRUE) # for model fitting

# spTimer
#set.seed(11)
out1<-spT.Gibbs(formula=y~1,data=fit,coords=~Longitude+Latitude,nItr=nItr,nBurn=nBurn,distance.method="euclidean",spatial.decay=spT.decay(distribution=Gamm(2,1),tuning=0.1))
pred1<-predict(out1,newdata=val,newcoords=~Longitude+Latitude)

# spBayes
out2<-run.spBayes(nItr=nItr,data=fit)
out2<- spRecover(out2, verbose=FALSE)
pred.coords<-unique(cbind(val$Longitude,val$Latitude))
x1 <- matrix(1,length(s),dim(val)[[1]]/length(s))
xx<-NULL
for(i in 1:dim(x1)[[2]]){
      xx[[i]]<-as.matrix(x1[,i]) 
}
x.mv<-mkMvX(xx)
pred2 <- spPredict(out2, pred.covars=x.mv, pred.coords=pred.coords)

# GAM
out3<-gam(formula=y~s(Longitude,Latitude),data=fit)
pred3<-predict(out3,newdata=val)

# validation statistics
val.spTimer<-spT.validation(val$y,pred1$Median)
val.spBayes<-spT.validation(val$y,apply(pred2[[1]], 1, median))
val.gam<-spT.validation(val$y,c(pred3))

val.spTimer
val.spBayes
val.gam

rm(out1); rm(out2); rm(out3)

## Non-grid data
pause()

## Non-grid data

n<-7
time<-5 # takes 5, and 60

#set.seed(22)
dat<-data.sim.GP.nogrid(nn=n*n,T=time,beta=c(5.0))

# Model fitting
#set.seed(11)
s<-sample(1:(n*n),n*n*0.2)
val<-spT.subset(data=dat, var.name="s.index", s = s, reverse = FALSE) # for model validation
fit<-spT.subset(data=dat, var.name="s.index", s = s, reverse = TRUE) # for model fitting

# spTimer
#set.seed(11)
out1<-spT.Gibbs(formula=y~1,data=fit,coords=~Longitude+Latitude,nItr=nItr,nBurn=nBurn,distance.method="euclidean",spatial.decay=spT.decay(distribution=Gamm(2,1),tuning=0.0001))
pred1<-predict(out1,newdata=val,newcoords=~Longitude+Latitude)

# spBayes
out2<-run.spBayes(nItr=nItr,data=fit)
out2<- spRecover(out2, start=nBurn, verbose=FALSE)
pred.coords<-unique(cbind(val$Longitude,val$Latitude))
x1 <- matrix(1,length(s),dim(val)[[1]]/length(s))
xx<-NULL
for(i in 1:dim(x1)[[2]]){
      xx[[i]]<-as.matrix(x1[,i]) 
}
x.mv<-mkMvX(xx)
pred2 <- spPredict(out2, pred.covars=x.mv, pred.coords=pred.coords)


# GAM
out3<-gam(formula=y~s(Longitude,Latitude),data=fit)
pred3<-predict(out3,newdata=val)

# validation statistics
val.spTimer<-spT.validation(pred1$Median,val$y)
val.spBayes<-spT.validation(apply(pred2[[1]], 1, median),val$y)
val.gam<-spT.validation(c(pred3),val$y)

val.spTimer
val.spBayes
val.gam

rm(out1); rm(out2); rm(out3)

#######################################################################################
#######################################################################################
#######################################################################################


#######################################################################################
################################# Real life Example  ##################################
#######################################################################################

pause()

# load libraries
library("spTimer");library("maps")

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
#dev.new();
map(database="state",regions="new york")
points(coords,pch=19,col=3)
points(coords,pch=1,col=1)
points(pred.coords,pch=3,col=4)
legend(x=-77.5,y=41.5,col=c(3,4),pch=c(19,3),cex=0.8,legend=c("Fitted sites","Validation sites"))
#dev.off()


# Fit GP model 
set.seed(11)
post.gp <- spT.Gibbs(formula=o8hrmax ~cMAXTMP+WDSP+RH, data=DataFit, 
        model="GP", coords=~Longitude+Latitude, scale.transform="SQRT",
        spatial.decay=spT.decay(distribution=Gamm(2,1),tuning=0.1))
print(post.gp)
summary(post.gp)
#plot(post.gp)

# Spatial prediction for the GP model
set.seed(11)
pred.gp <- predict(post.gp, newdata=DataValPred, newcoords=~Longitude+Latitude)
print(pred.gp)
names(pred.gp)
# validation criteria
spT.validation(DataValPred$o8hrmax,c(pred.gp$Median))  

###############################
## For Figures 8 (a) -- (d)  ##
## Press Enter:              ##
###############################
pause()
# Figures 8 (a) -- (d)
data(NYgrid)
set.seed(11)
post.gp2 <- spT.Gibbs(formula=o8hrmax ~cMAXTMP+WDSP+RH,   
        data=NYdata, model="GP", coords=~Longitude+Latitude, 
        scale.transform="SQRT", nItr=1500, nBurn=0000,
	  spatial.decay=spT.decay(distribution=Gamm(2,1),tuning=0.1))
par(mfrow=c(2,2))
plot(post.gp2$betap[1,],type="l",main="Intercept",xlab="Iterations",ylab="",ylim=c(-1,6))
plot(post.gp2$betap[2,],type="l",main="cMAXTMP",xlab="Iterations",ylab="",ylim=c(0.06,.25))
plot(post.gp2$betap[3,],type="l",main="WDSP",xlab="Iterations",ylab="",ylim=c(-0.01,0.25))
plot(post.gp2$betap[4,],type="l",main="RH",xlab="Iterations",ylab="",ylim=c(-.4,.33))
par(mfrow=c(1,1))

pause()

		
###############################
## For Figures 9 (a) and (b) ##
## Press Enter:              ##
###############################
pause()

# Figures 9 (a) and (b)
# Predict on grids
data(NYgrid)
set.seed(11)
post.gp2 <- spT.Gibbs(formula=o8hrmax ~cMAXTMP+WDSP+RH,   
        data=NYdata, model="GP", coords=~Longitude+Latitude, 
        scale.transform="SQRT",spatial.decay=spT.decay(distribution=Gamm(2,1),tuning=0.1))
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
	surf<-fnc.delete.map.XYZ(xyz=surf)
	#map(database="state",regions="new york")
	image.plot(surf, xlab="Longitude",ylab="Latitude",axes=F, ...)
	contour(surf,nlevels=10,lty=3,add=T)
	map(database="state",regions="new york",add=T)
	axis(1);axis(2)
}

# Section 5: code for Figure 8(a)
# prediction for day 60
day<-60
surfplot(day, val=grid.val, col = rainbow_hcl(100, start = 200, end = 0))
text(coords,labels=round(true.val[day,],1),cex=0.8,col=1)

# Section 5: code for Figure *(b)
# sd for day 60
# Press Enter:
pause()

# Section 5: code for Figure 8(b)
# sd for day 60
day<-60
surfplot(day, val=grid.sd,col = diverge_hcl(100, h = c(246, 40), c = 96, l = c(65, 90)))
points(coords,pch=19,cex=1,col=2)
points(coords,pch=1,cex=1,col=1)



##################### Comparison with GAM for real-life data  #########################
#######################################################################################

pause()
data(NYdata)
s<-c(8,11,12,14,18,21,24,28)
DataFit<-spT.subset(data=NYdata, var.name=c("s.index"), s=s, reverse=TRUE) 
DataFit<-subset(DataFit, with(DataFit, !(Day %in% c(30, 31) & Month == 8)))
DataValPred<-spT.subset(data=NYdata, var.name=c("s.index"), s=s) 
DataValPred<-subset(DataValPred, with(DataValPred, !(Day %in% c(30, 31) & Month == 8)))

# GAM model
library(mgcv)
set.seed(11)
fit.gam <- gam(o8hrmax ~ s(cMAXTMP) + s(WDSP) + s(RH) + s(Longitude, Latitude, k=10), data = DataFit)
pred.gam <- predict(fit.gam,DataValPred, interval="prediction")
spT.validation(DataValPred$o8hrmax,pred.gam)

# spTimer model
set.seed(11)
post.gp <- spT.Gibbs(formula=o8hrmax ~cMAXTMP+WDSP+RH,   
        data=DataFit, model="GP", coords=~Longitude+Latitude, 
        scale.transform="SQRT",spatial.decay=spT.decay(distribution=Gamm(2,1),tuning=0.1))
set.seed(11)
pred.gp <- predict(post.gp, newdata=DataValPred, newcoords=~Longitude+Latitude)
spT.validation(DataValPred$o8hrmax,c(pred.gp$Median))  


########################### Run demo file for NY example  #############################
#######################################################################################

## run demo(nyExample) file?
pause()

demo(nyExample)


#####################                  The End                #########################
#######################################################################################

pause()


#######################################################################################
#######################################################################################
#######################################################################################

