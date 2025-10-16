# ------------------------------------------
# -- An example for Confirmatory Analysis --
# ------------------------------------------
library(GVEMGRM)
attach(toy_data) # load true parameters & graded response data y

# -- initial parameters --
# -- note that, the initial value of B, Sig and variational parameters are unnecessary
N <- nrow(y)
J <- ncol(y)
K <- 2

model  <- (true_A!=0)*1
init_A <- matrix(.01, nrow=J, ncol=K); init_A[model==0] <- 0

# -- call gvemgrm --
output <- gvemgrm(y = y, init_A = init_A, tar_mod = model)
names(output)
output$new_A


# ------------------------------------------
# -- An example for Exploratory Analysis ---
# ------------------------------------------

# -- note that, the key for exploratory analysis is
# -- 1. set the target model as full model.
# -- 2. keep Sigma as identity matrix, i.e., set init_Sig = diag(1,K) and is_sigmaknown = 1. 
# -- 3. to avoid numerical error, set random value for initial value of A.
# -- factor rotation technique may cause the column swapping.
library(GVEMGRM)
library(GPArotation)
attach(toy_data)

# -- initial parameters --
N <- nrow(y)
J <- ncol(y)
K <- 2

set.seed(627)
model  <- matrix(1,J,K)
init_A <- matrix(runif(J*K,.01,.02), nrow=J, ncol=K)
init_Sig <- diag(1,K)

# -- call gvemgrm --
output <- gvemgrm(y = y, init_A = init_A, tar_mod = model, is_sigmaknown = 1)
esti_A <- output$new_A
fa_rot <- quartimin(esti_A) # oblique rotation
rot_A   <- fa_rot$loadings
rot_Sig <- fa_rot$Phi
cut_A <- rot_A
cut_A[abs(rot_A)<.3] <- 0

rot_A
cut_A
rot_Sig


