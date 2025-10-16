#' IW-GVEM for MGRM
#' 
#' @description
#' Importance-weighted Gaussian variational expectation maximization algorithm 
#' for estimating item parameters in multidimensional graded response model.
#' 
#' @param y N*J mat, graded item responses.
#' @param init_A   J*K mat, initial value of A, a good choice is estimated A from GVEM.
#' @param init_B   J*R vec, initial value of B, a good choice is estimated B from GVEM.
#' @param init_Sig K*K mat, initial value of \eqn{\Sigma}, a good choice is estimated \eqn{\Sigma} from GVEM.
#' @param mu_n    N*K mat, parameter \eqn{\mu_i} of Gaussian variational distribution \eqn{q_i(\theta_i)}, obtain from GVEM.
#' @param sigma_n K*K*N array, parameter \eqn{\Sigma_i} of Gaussian variational distribution \eqn{q_i(\theta_i)}, obtain from GVEM.
#' @param is_sigmaknown default is 0, meaning \eqn{\Sigma} is unknown and will be estimated; 1 means \eqn{\Sigma} is known and fixed.
#' @param seed random number seed, default = 627.
#' @param S number of Monte Carlo samples.
#' @param M importance weight size.
#' @param beta1 exponential decay rates of the exponential moving averages of gradient, default is 0.9.
#' @param beta2 exponential decay rates of the exponential moving averages of squared gradient, default is 0.999.
#' @param eta_A learning rate of A.
#' @param eta_gam learning rate of \eqn{\gamma}.
#' @param eta_sig learning rate of \eqn{\Sigma}.
#' @param eps     a small positive value to ensure numerical stability, default = .001.
#' @param max_iter maximum number of iterations. default is 100.
#' @param tol.para tolerance for convergence of parameters, default is 1e-3.
#' @return A list contains:
#'   \tabular{ll}{
#'     \code{new_A} \tab estimated A (slope parameters).\cr
#'     \code{new_B} \tab estimated B (threshold parameters).\cr
#'     \code{new_Sig} \tab estimated \eqn{\Sigma} (correlation of latent traits).\cr
#'     \code{n2lb} \tab negative twice the evidence lower bound (ELBO).\cr
#'     \code{t} \tab number of iterations.\cr
#'     \code{cpu_time} \tab computing time.\cr
#'   }
#' @export
#' @example inst/examples/example_for_iwgvem.R

iwgvemgrm <- function(
    y,
    init_A,
    init_B,
    init_Sig,
    mu_n,
    sigma_n,
    is_sigmaknown = 0,
    seed = NULL,
    S = 10,
    M = 10,
    beta1 = 0.9,
    beta2 = 0.999,
    eta_A   = 0.05,
    eta_gam = 0.005,
    eta_sig = 0.005,
    eps   = 0.001,
    max_iter = 100,
    tol.para = 1e-3
){
  
  if(!is.null(seed)){set.seed(seed)}
  R <- as.vector(apply(X = y, MARGIN = 2, FUN = function(x){length(unique(x))}))
  Mod <- (init_A!=0)*1
  
  cpu_time <- proc.time()
  output <- rcpp_iwgvemgrm(y = y, R = R, old_A = init_A, old_B = init_B, old_sig = init_Sig,
                       mu_n = mu_n, sigma_n = sigma_n, Mod = Mod,
                       is_sigmaknown = is_sigmaknown, S = S, M = M,
                       beta1 = beta1, beta2 = beta2, eta_A = eta_A, eta_gam = eta_gam,
                       eta_sig = eta_sig, eps = eps,
                       max_iter = max_iter, tol_para = tol.para)
  cpu_time <- proc.time() - cpu_time
  cpu_time <- as.numeric(cpu_time)[3]
  
  result <- list(
    new_A    = output$new_A,
    new_B    = output$new_B,
    new_sig  = output$new_sig,
    n2lb     = output$n2lb,
    t        = output$t,
    cpu_time = cpu_time
  )
  return(result)
}

