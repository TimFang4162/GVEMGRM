library(GVEMGRM)
set.seed(627)
true_A <- matrix(0,12,2)
true_A[ 1:6,1] <- seq(2.5,1.5,-0.2)
true_A[7:12,2] <- seq(2.4,1.4,-0.2)
true_B <- matrix(NA,12,4)
true_B[1:6,1] <- 0 # first 6 items are binary
true_B[7:12,] <- rep(c(-1.2,-0.4,0.4,1.2),each=6) # last 6 items are 5-point scale
true_Sig <- matrix(c(1,0.2,0.2,1),2,2)
toy_data <- simudata(200, true_A, true_B, true_Sig)
theta    <- toy_data$theta
y        <- toy_data$y

# toy_data$true_A   <- true_A
# toy_data$true_B   <- true_B
# toy_data$true_Sig <- true_Sig
# save(toy_data, file = "./data/toy_data.rda")
# rm(list = ls())
# attach(toy_data)
