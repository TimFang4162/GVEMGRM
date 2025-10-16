# ------------------------------------------
# -- An example for Confirmatory Analysis --
# ------------------------------------------

library(GVEMGRM)
attach(toy_data)

# -- initial parameters --
# -- note that, the initial value of B, Sig and variational parameters are unnecessary
N <- nrow(y)
J <- ncol(y)
K <- 2

model  <- (true_A!=0)*1
init_A <- matrix(.01, nrow=J, ncol=K); init_A[model==0] <- 0

# -- call gvemgrm first and then iwgvemgrm --
gvem_output <- gvemgrm(y = y, init_A = init_A, tar_mod = model)

iwgvem_output <- iwgvemgrm(
  y = y, init_A = gvem_output$new_A, init_B = gvem_output$new_B, init_Sig = gvem_output$new_sig,
  mu_n = gvem_output$mu_n, sigma_n = gvem_output$sigma_n)
names(iwgvem_output)

