#' Generate simulation data
#'
#' @param N sample size, i.e., number of examinees.
#' @param A J*K mat, slope parameters.
#' @param B J*max(R) mat, threshold parameters for each item. 
#' @param Sig K*K mat, correlation matrix of latent traits.
#' @param R J*1 vec, number of grade for each item.
#'
#' @return A list containing:
#'   \tabular{ll}{
#'     \code{theta} \tab individual scores with dimension N*K.\cr
#'     \code{y} \tab graded response with dimension N*J.\cr
#'   }
#' @export
#' @importFrom MASS, mvrnorm
#' @importFrom stats, plogis, rmultinom
#' @example inst/examples/example_for_simudata.R
simudata <- function(N, A, B, Sig, R=NULL){
  
  J <- nrow(A)
  K <- ncol(A)
  
  if(is.null(R))  R <- rowSums(!is.na(B)) + 1
  
  x <- mvrnorm(n=N, mu=rep(0,K), Sigma = Sig)
  y <- matrix(NA, nrow=N, ncol=J)
  f <- matrix(0, nrow=N, ncol=max(R)+1)
  
  min_freq <- .02  # minimum frequency of each grade for each item.
  
  for(j in 1:J){
    
    freq <- 0
    while(freq < min_freq){
      Rj <- R[j]
      Aj <- A[j,,drop=FALSE]
      Bj <- B[j,1:(Rj-1),drop=FALSE]
      
      logitf <- matrix(x%*%t(Aj),nrow=N,ncol=Rj-1,byrow=FALSE) - matrix(Bj,nrow=N,ncol=Rj-1,byrow=TRUE)
      f[,]     <- NA
      f[,1]    <- 1
      f[,Rj+1] <- 0
      f[,2:Rj] <- plogis(logitf)
      p <- f[,1:Rj] - f[,(1:Rj)+1]
      
      yj__ <- apply(X=p, MARGIN=1, FUN=rmultinom, n=1, size=1)
      yj   <- apply(X=yj__, MARGIN=2, FUN=which.max) - 1
      
      freq <- min(table(yj))/N
    }
    y[,j] <- yj
  }
  
  storage.mode(y) <- "integer"
  
  return(list(theta = x, y = y))
  
}

