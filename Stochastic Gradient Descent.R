library(rgr)
library(Deriv)
library(ggplot2)

## Function to calculate gradient of function by sig
calcSigGrad <- function(x,sig,mu) {
  return(eval(D(expr,'sig')))
}

## Function to calculate gradient of function by mu
calcMuGrad <- function(x,sig,mu) {
  return(eval(D(expr,'mu')))
}

## Function to calculate function value for given values.
## This value is used in convergence check
calcFunVal <- function(x,sig,mu) {
  costFn <- expression (log(1/sqrt(2*pi))+log(1/sig)+(-((x-mu)^2)/(2*(sig^2))))
  return(eval(costFn))
}

## Main function that performs stochastic gradient ascent
calcStochGrad <- function(mu.init,sig.init,pts,rate,epoch.max)
{
  print("********************************************************")
  print(paste0("Started gradient algorithm for learning rate: ", rate))
  print("********************************************************")
  
  ## Initialize values
  sig <- sig.init
  mu <- mu.init
  cost.prev <- 0
  cost <- 0
  is.converged <- FALSE
  epoch <- 1
  i <- 1
  
  ## Create a matrix to hold iteration,change values
  res <- matrix(,nrow=epoch.max,ncol=4)
  
  iter.count <- length(pts$n_tokens_content)
  
  ## Iterate through entire dataset by randomizing input dataset
  ## and iterate until convergence is met 
  while(epoch <= epoch.max)
  {
    i <- 1
    
    ## Randomize the input dataset
    readData <- pts[sample(1:nrow(pts)), ]
    ipValue <- readData$n_tokens_content
    
    ## For each value from dataset, update sigma and mu
    while(i <= iter.count) {
      sig <- sig + (rate*calcSigGrad(ipValue[i],sig,mu))
      mu <- mu + (rate*calcMuGrad(ipValue[i],sig,mu))
      i <- i+1
    }
    
    cost <- calcFunVal(ipValue[i],sig,mu)
    
    ## Ignore change for first iteration as we dont have old value
    ## For other iterations, store iteration number and change values
    if (!(epoch==1)) {
      cost.delta <- cost-cost.prev
      sig.delta <- sig-sig.prev
      mu.delta <- mu-mu.prev
      res[epoch-1, ] <- c(epoch-1,mu.delta,sig.delta,cost.delta)
      
      ## Convergence check
      if ((round(mu.delta,3)==0) & (round(sig.delta,3))==0) {
        print(paste0("Converged for ",rate, " in ", epoch, " iterations"))
        print(paste0("value of mu : ", mu))
        print(paste0("value of sig : ", sig))
        break
      }
      
    }
    sig.prev <- sig
    mu.prev <- mu
    cost.prev <- cost
    epoch <- epoch+1
    
  }
  
  ## Omit NA values as convergence could be met much before!
  return(res[rowSums(is.na(res))!=ncol(res), ])
}

start.time <- proc.time()

## Input funtion to be optimized
expr <- expression (log(1/sqrt(2*pi))+log(1/sig)+(-((x-mu)^2)/(2*(sig^2))))

## Read input file into dataframe
inputData <- read.csv("OnlineNewsPopularity.csv")
ipData <- inputData$n_tokens_content

## Call SG algorithm for 3 iterations
result1 <- calcStochGrad(400,300,inputData,0.1,1000)
result3 <- calcStochGrad(400,300,inputData,0.3,1000)
result5 <- calcStochGrad(400,300,inputData,0.5,1000)

## Plot Iteration Vs change in mu and sig for the various learning rates
## in same graph

print("Plotting data for 3 iterations..")

## Plot in Q1.2 (Change vs Iteration)
png(file = "Q2.2.jpg")

## Define Plot Title
plotTitle <- "Iteration vs change in mu/sigma"

## Plot the iteration vs change in sigma and change mu in single graph
plot(result1[,1],result1[,2],col="red",pch=19,xlab="Iteration number", ylab="Change in mu/sig")

legend("topright", 
       legend = c("0.1", "0.3","0.5"), 
       col = c("red","blue","green"),
       lty=c(1,1,1)) 

legend(380,4, 
       legend = c("mu", "sigma"), 
       col = c("black"),
       pch = c(19,23)) 

points(result1[,1],result1[,3],col="red",pch=23)

## Add the iteration vs change in sigma and change mu to above graph
points(result3[,1],result3[,2],col="blue",pch=19)
points(result3[,1],result3[,3],col="blue",pch=23)

## Add the iteration vs change in sigma and change mu to above graph
points(result5[,1],result5[,2],col="green",pch=19)
points(result5[,1],result5[,3],col="green",pch=23)

# Save the file.
dev.off()

## Plot for cost function

print("Plotting data for 3 iterations..")

## Plot in Q1.2 (Change vs Iteration)
png(file = "Q2.2_1.jpg")

## Define Plot Title
plotTitle <- "Iteration vs change in cost function"

## Plot the iteration vs change in sigma and change mu in single graph
plot(result1[,1],result1[,4],col="red",pch=19,xlab="Iteration number", ylab="Change in mu/sig")

legend("topright", 
       legend = c("0.1", "0.3","0.5"), 
       col = c("red","blue","green"),
       lty=c(1,1,1)) 

## Add the iteration vs change in sigma and change mu to above graph
points(result3[,1],result3[,4],col="blue",pch=19)

## Add the iteration vs change in sigma and change mu to above graph
points(result5[,1],result5[,4],col="green",pch=19)

# Save the file.
dev.off()

## Q2.3 : Plot Histogram of data

png(file = "Q2.4.jpg")

## Plot a histogram
qplot(inputData$n_tokens_content,geom="histogram",binwidth=100,xlim=c(-100,5000),fill=I("grey"),col=I("red"),main="Histogram for n_tokens_content",xlab="n_tokens_content")

# Save the file.
dev.off()

print("Code completed. Below are run time statistics:")

end.time <- proc.time()
print(end.time-start.time)
