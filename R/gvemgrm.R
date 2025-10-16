#' GVEM algorithm for MGRM
#' 
#' @description
#' Gaussian variational expectation maximization algorithm for estimating 
#' item parameters in multidimensional graded response model.
#' @param y N*J mat, graded item responses.
#' @param init_A J*K mat, initial value of A.
#' @param init_B J*R vec, initial value of B.
#' @param init_sig K*K mat, initial value of \eqn{\Sigma}.
#' @param init_ksi1 N*J mat, initial value of \eqn{\xi_{1}}, the variational parameter.
#' @param init_ksi2 N*J mat, initial value of \eqn{\xi_{2}}, the variational parameter.
#' @param tar_mod J*K mat, target model (structure of A), the element only 0 or 1 is valid.
#' @param is_sigmaknown default is 0, meaning \eqn{\Sigma} is unknown and will be estimated; 1 means \eqn{\Sigma} is known and fixed.
#' @param maxiter maximum number of iterations. default is 100.
#' @param tol_n2vlb tolerance for convergence of the variational lower bound, default is 1e-4.
#' @param tol_para tolerance for convergence of parameters, default is 1e-3.
#' @param stop_cri stopping criterion: 1 for variational lower bound, 2 for parameter changes.
#' @param calcu_n2vlb whether to calculate the variational lower bound in each iteration when stop.cri = 2, default is 0.
#' @return A list contains:
#'   \tabular{ll}{
#'     \code{new_A} \tab estimated A (slope parameters).\cr
#'     \code{new_B} \tab estimated B (threshold parameters).\cr
#'     \code{new_Sig} \tab estimated \eqn{\Sigma} (correlation of latent traits).\cr
#'     \code{new_ksi1} \tab variational parameters \eqn{\xi_1}.\cr
#'     \code{new_ksi2} \tab variational parameters \eqn{\xi_2}.\cr
#'     \code{mu_n} \tab parameter \eqn{\mu_i} of Gaussian variational distribution \eqn{q_i(\theta_i)} for all \eqn{i=1,\ldots,N}.\cr
#'     \code{sigma_n} \tab parameter \eqn{\Sigma_i} of Gaussian variational distribution \eqn{q_i(\theta_i)} for all \eqn{i=1,\ldots,N}.\cr
#'     \code{n2vlb} \tab negative twice the evidence lower bound (ELBO).\cr
#'     \code{vbic} \tab variational BIC.\cr
#'     \code{n2vlb_seq} \tab negative twice the ELBO at each iteration, returns NA when stop_cri = 2 and calcu_n2vlb = 0.\cr
#'     \code{iter} \tab number of iterations.\cr
#'     \code{converge_n2vlb} \tab indicates ELBO convergence status for stop_cri = 1, 1 if converged, 0 otherwise.\cr
#'     \code{converge_para} \tab indicates parameters convergence status for stop_cri = 2, 1 if converged, 0 otherwise.\cr
#'     \code{cpu_time} \tab computing time.\cr
#'   }
#' @export
#' @example inst/examples/example_for_gvem.R

gvemgrm <- function(
  y,
  init_A,
  init_B    = NULL,
  init_sig  = NULL,
  init_ksi1 = NULL,
  init_ksi2 = NULL,
  tar_mod   = NULL,
  is_sigmaknown = 0,
  maxiter   = 200,
  tol_n2vlb = 1e-4,
  tol_para  = 1e-3,
  stop_cri  = 2,
  calcu_n2vlb = 0
  
){
  
  if(!is.integer(y)){
    storage.mode(y) <- "integer"
  }
  R <- as.vector(apply(X = y, MARGIN = 2, FUN = function(x){length(unique(x))}))
  if(is.null(init_B)){
    init_B <- matrix(NA, nrow=ncol(y), ncol=max(R)-1)
    for(j in 1:ncol(y)){
      init_B[j,1:(R[j]-1)] <- qlogis(cumsum(table(y[,j]))/nrow(y))[1:(R[j]-1)]
    }
  }
  if(is.null(init_sig)){
    init_sig <- diag(ncol(init_A))
  }
  if(is.null(init_ksi1)){
    init_ksi1 <- matrix(0.05, nrow=nrow(y), ncol=ncol(y))
  }
  if(is.null(init_ksi2)){
    init_ksi2 <- matrix(0.05, nrow=nrow(y), ncol=ncol(y))
  }
  if(is.null(tar_mod)){
    tar_mod <- (init_A!=0)*1
  }
  if(!is.integer(tar_mod)){
    storage.mode(tar_mod) <- "integer"
  }
  
  cpu_time <- proc.time()
  
  output <- rcpp_gvemgrm(y = y, R = R, old_A = init_A, old_B = init_B, old_sig = init_sig,
                         old_ksi1 = init_ksi1, old_ksi2 = init_ksi2, Mod = tar_mod,
                         is_sigmaknown = is_sigmaknown, max_iter = maxiter,
                         tol_n2vlb = tol_n2vlb, tol_para = tol_para,
                         stop_cri = stop_cri, is_calcu_n2vlb = calcu_n2vlb)
  
  cpu_time <- proc.time() - cpu_time
  cpu_time <- as.numeric(cpu_time)[3]
  
  vbic <- output$n2vlb + log(nrow(y))*(sum(output$new_A!=0) + sum(output$new_B!=0,na.rm=T) + (sum(output$new_sig!=0) - ncol(output$new_sig))/2) 
  if(stop_cri == 2 & calcu_n2vlb == 0){output$n2vlb_seq <- NA}
  
  # ---- return output ----
  result <- list(
    new_A    = output$new_A,
    new_B    = output$new_B,
    new_sig  = output$new_sig,
    new_ksi1 = output$new_ksi1,
    new_ksi2 = output$new_ksi2,
    mu_n     = output$mu_n,
    sigma_n  = output$sigma_n,
    n2vlb    = output$n2vlb, # negative 2 variational lower bound
    vbic     = vbic,
    n2vlb_seq= output$n2vlb_seq,
    iter     = output$it,
    converge_n2vlb = output$converge_n2vlb,
    converge_para  = output$converge_para,
    cpu_time       = cpu_time
  )
  
  return(result)  
}
