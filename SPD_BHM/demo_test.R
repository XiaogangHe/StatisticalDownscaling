################################################################
###################  New York data example #####################
################################################################

# define a pause function
pause <- function() invisible(readline("\n Please Enter to continue: \n"))

############################################
## Model spatial only data using GP model ##
############################################
pause()


# spatial only data 
# we use meuse data from sp package
library("spTimer")
library("akima")
library(sp)
data(meuse)
# model with GP
set.seed(11)
post.gp <- spT.Gibbs(formula=zinc ~ sqrt(dist),   
    data=meuse, model="GP", coords=~x+y, nItr=500, nBurn=100,
	spatial.decay=spT.decay(distribution=Gamm(2,1), tuning=0.5),
	distance.method="euclidean",scale.transform="LOG")
summary(post.gp)

plot(post.gp, surface="Mean", title=FALSE)


