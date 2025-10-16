#define RCPP_ARMADILLO_RETURN_ANYVEC_AS_VECTOR
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;


arma::mat b2gamma(const arma::mat &B,
                  const arma::Col<int> &R){
  
  int j;
  int J = B.n_rows;
  arma::mat gamma = B;
  for(j=0;j<J;j++){
    if(R(j)>2){
      gamma.row(j).subvec(1,R(j)-2) = log(arma::diff(B.row(j).subvec(0,R(j)-2)));
    }
  }
  return gamma;
}


arma::mat gamma2b(const arma::mat &gamma,
                  const arma::Col<int> &R){
  
  int j;
  int J = gamma.n_rows;
  arma::mat B = gamma;
  for(j=0;j<J;j++){
    if(R(j)>2){
      B.row(j).subvec(1,R(j)-2) = exp(B.row(j).subvec(1,R(j)-2));
      B.row(j).subvec(0,R(j)-2) = arma::cumsum(B.row(j).subvec(0,R(j)-2));
    }
  }
  return B;
}


void calcu_Fjr(arma::mat &Fjr,
               const arma::mat &Theta_is,
               const arma::mat &A,
               const arma::mat &B,
               const arma::Row<int> &y_i
){
  // for certain i, s, calculate F(a_{j}^{T}\theta_{i}^{(s,m)} - b_{jr}) for all
  //   m = 1,...,M and j = 1,...,J with y_{ij}.
  
  int J = y_i.n_elem;
  Fjr.ones();
  for(int j=0;j<J;j++){
    if(y_i(j)>0){
      Fjr.col(j) = 1/(1 + exp(B(j,y_i(j)-1)-Theta_is*arma::trans(A.row(j))));
    }
  }
}


void calcu_Fjrp(arma::mat &Fjrp,
                const arma::mat &Theta_is,
                const arma::mat &A,
                const arma::mat &B,
                const arma::Row<int> &y_i,
                const arma::Col<int> &R
){
  // for certain i, s, calculate F(a_{j}^{T}\theta_{i}^{(s,m)} - b_{jr}) for all
  //   m = 1,...,M and j = 1,...,J with y_{ij} + 1.
  
  int J = y_i.n_elem;
  Fjrp.zeros();
  for(int j=0;j<J;j++){
    if(y_i(j)<R(j)-1){
      Fjrp.col(j) = 1/(1 + exp(B(j,y_i(j))-Theta_is*arma::trans(A.row(j))));
    }
  }
}


void calcu_tid_wis(arma::vec &tid_wis,
                   const arma::mat &Theta_is,
                   const arma::mat &Theta_is__,
                   const arma::rowvec &mu_i,
                   const arma::mat &sigma_i,
                   const arma::mat &sig,
                   const arma::mat &Fjr,
                   const arma::mat &Fjrp
){
  
  arma::vec log_wis = 0.5*arma::log_det_sympd(sigma_i) + 0.5*arma::sum((Theta_is__*arma::inv(sigma_i))%Theta_is__,1);
  log_wis -= 0.5*arma::log_det_sympd(sig) + 0.5*arma::sum((Theta_is*arma::inv(sig))%Theta_is,1); 
  log_wis += sum(log(Fjr - Fjrp), 1);
  tid_wis = exp(log_wis);
  tid_wis /= sum(tid_wis);
  
}


arma::mat calcu_grad_sig(const arma::mat &old_sig,
                         const arma::mat &Theta_is,
                         const arma::vec &tid_wis
){
  arma::mat temp = arma::inv(old_sig);
  arma::mat grad_sig = 0.5*( - temp + temp * (arma::trans(Theta_is)*(Theta_is.each_col()%tid_wis)) * temp );
  return grad_sig;
}


arma::mat calcu_grad_A(const arma::Mat<int> &M,
                       const arma::mat &Theta_is,
                       const arma::vec &tid_wis,
                       const arma::mat &Fjr,
                       const arma::mat &Fjrp
){
  
  arma::mat temp   = 1-Fjr-Fjrp;
  arma::mat grad_A = arma::trans(temp.each_col()%tid_wis)*Theta_is;
  grad_A = grad_A % M;
  return grad_A;
}


arma::mat calcu_grad_gam(const arma::mat &old_gam,
                         const arma::mat &Theta_is,
                         const arma::vec &tid_wis,
                         const arma::mat &Fjr,
                         const arma::mat &Fjrp,
                         const arma::Row<int> &y_i,
                         const arma::Col<int> &R){
  
  int j;
  int J = old_gam.n_rows;
  arma::mat grad_gam = arma::zeros(J,max(R)-1);
  arma::mat temp = 1-Fjr-Fjrp;
  grad_gam.col(0) = - sum(temp.each_col()%tid_wis,0).t();
  for(j=0;j<J;j++){
    if(R(j)>2){
      for(int r=2;r<=R(j)-1;r++){
        if(y_i(j)==r-1){
          grad_gam(j,r-1) = sum(Fjrp.col(j)%(1-Fjrp.col(j))/(Fjr.col(j)-Fjrp.col(j))%tid_wis)*exp(old_gam(j,r-1));
        }
        else if(y_i(j)>=r){
          grad_gam(j,r-1) = - sum((1-Fjr.col(j)-Fjrp.col(j))%tid_wis)*exp(old_gam(j,r-1));
        }
      }
    }
  }
  
  return grad_gam;
}


arma::mat update_param(const arma::mat &old_param,
                       const arma::mat &grad_param,
                       arma::mat &vu_param,
                       arma::mat &r_param,
                       const double beta1,
                       const double beta2,
                       const double beta1t,
                       const double beta2t,
                       const double eps,
                       const double eta){
  
  
  vu_param = beta1*vu_param + (1-beta1)*grad_param;
  r_param  = beta2*r_param  + (1-beta2)*(grad_param%grad_param);
  vu_param = vu_param/(1-beta1t);
  r_param  = r_param /(1-beta2t);
  
  arma::mat hat_grad_param = eta*vu_param/(sqrt(r_param)+eps);
  arma::mat new_param      = old_param + hat_grad_param;
  
  return new_param;
}


arma::mat gen_theta_is(const arma::rowvec &mu,
                       const arma::mat &sigma,
                       int M){
  
  return arma::trans(arma::mvnrnd(mu.t(), sigma, M));
}


double calcu_n2lb(const arma::Mat<int> &y,
                  const arma::mat &new_A,
                  const arma::mat &new_B,
                  const arma::mat &new_sig,
                  const arma::mat  &mu_n,
                  const arma::cube &sigma_n,
                  arma::mat &Fjr,
                  arma::mat &Fjrp,
                  const int N,
                  const int K,
                  const int S,
                  const int M,
                  const arma::Col<int> &R){
  
  double lb = 0.0;
  arma::mat Theta_is  (M,K);
  arma::mat Theta_is__(M,K);
  arma::vec log_wis(M);
  arma::mat sigma_i(K,K);
  arma::rowvec mu_i(K);
  int i, s;
  for(i=0;i<N;i++){
    
    mu_i    = mu_n.row(i);
    sigma_i = sigma_n.slice(i);
    
    for(s=0;s<S;s++){
      
      Theta_is   = arma::trans(arma::mvnrnd(mu_i.t(), sigma_i, M));
      Theta_is__ = Theta_is.each_row() - mu_i;
      
      calcu_Fjr (Fjr,  Theta_is, new_A, new_B, y.row(i));
      calcu_Fjrp(Fjrp, Theta_is, new_A, new_B, y.row(i), R);
      
      log_wis = 0.5*arma::log_det_sympd(sigma_i) + 0.5*arma::sum((Theta_is__*arma::inv(sigma_i))%Theta_is__,1);
      log_wis -= 0.5*arma::log_det_sympd(new_sig) + 0.5*arma::sum((Theta_is*arma::inv(new_sig))%Theta_is,1); 
      log_wis += sum(log(Fjr - Fjrp), 1);
      
      lb += 1.0 / S * log(1.0 / M * sum(exp(log_wis)));
    }
    
  }
  
  lb *= -2;
  
  return lb;
}


// [[Rcpp::export]]
Rcpp::List rcpp_iwgvemgrm(
    const arma::Mat<int> &y,   // N*J mat, observed item responses.
    const arma::Col<int> &R,
    arma::mat old_A,           // J*K mat, initial value of A, i.e., estimated A from GVEM.
    arma::mat old_B,           // J*R vec, initial value of B, i.e., estimated B from GVEM.
    arma::mat old_sig,         // K*K mat, initial value of \Sigma, i.e., estimated \Sigma from GVEM.
    const arma::mat  &mu_n,    // N*K mat, all mu_i of q_i(theta_i) obtained by GVEM.
    const arma::cube &sigma_n, // K*K*N array, all sigma_i of q_i(theta_i) obtained by GVEM.
    
    const arma::Mat<int> &Mod,
    const int is_sigmaknown,
    
    const int S = 10,
    const int M = 10,
    const double beta1 = 0.9,    // exponential decay rates
    const double beta2 = 0.999,  // exponential decay rates
    const double eta_A   = 0.05,     // learning rate for Adaptive moment estimation
    const double eta_gam = 0.01,     // learning rate for Adaptive moment estimation
    const double eta_sig = 0.005,    // learning rate for Adaptive moment estimation
    const double eps = 0.001,
    
    const int    max_iter = 100,
    const double tol_para = 1e-4

){
  
  int N = y.n_rows;
  int J = old_A.n_rows;
  int K = old_A.n_cols;
  int max_R = max(R);
  
  arma::mat old_gam = b2gamma(old_B, R);
  
  arma::mat new_A   = old_A;
  arma::mat new_B   = old_B;
  arma::mat new_sig = old_sig;
  arma::mat new_gam = old_gam;
  
  arma::mat old_B_nona = old_B; // replace na with 0 for calcu err B
  arma::mat new_B_nona = new_B; // replace na with 0 for calcu err B
  arma::vec error_para(3);
  
  // ---- initial value ----
  arma::mat vu_A   = arma::zeros(J,K);
  arma::mat vu_gam = arma::zeros(J,max_R-1);
  arma::mat vu_sig = arma::zeros(K,K);
  
  arma::mat r_A   = arma::zeros(J,K);
  arma::mat r_gam = arma::zeros(J,max_R-1);
  arma::mat r_sig = arma::zeros(K,K);
  
  double beta1t = 1.0;
  double beta2t = 1.0;
  
  // ---- working mat ----
  arma::mat Theta_is  (M,K);
  arma::mat Theta_is__(M,K);
  arma::mat Fjr (M,J);
  arma::mat Fjrp(M,J);
  arma::vec tid_wis(M);
  
  arma::mat grad_A(J,K);
  arma::mat grad_gam(J,max_R-1);
  arma::mat grad_sig(K,K);
  
  // ---- iteration ----
  int i, s;
  int t = 0;
  while(t < max_iter){
    
    t += 1;
    Rprintf("t: %03d\r", t);
    
    // -- reset gradient --
    grad_A.fill(0);
    grad_gam.fill(0);
    grad_sig.fill(0);
    
    for(i=0;i<N;i++){
      for(s=0;s<S;s++){
        
        // -- step 1: for certain i, s, draw M sample of theta from qi --
        Theta_is   = arma::trans(arma::mvnrnd(mu_n.row(i).t(), sigma_n.slice(i), M));
        Theta_is__ = Theta_is.each_row() - mu_n.row(i);
        
        // -- step 2: for certain i, s, calculate \tilde{w_{i}^{(s,m)}} for all m --
        calcu_Fjr(Fjr, Theta_is, old_A, old_B, y.row(i));
        calcu_Fjrp(Fjrp, Theta_is, old_A, old_B, y.row(i), R);
        calcu_tid_wis(tid_wis, Theta_is, Theta_is__, mu_n.row(i), sigma_n.slice(i),
                      old_sig, Fjr, Fjrp);
        
        // -- step 3: calculate gradient of A, B, Omega --
        grad_A   += 1.0/S * calcu_grad_A(Mod, Theta_is, tid_wis, Fjr, Fjrp);
        grad_gam += 1.0/S * calcu_grad_gam(old_gam, Theta_is, tid_wis, Fjr, Fjrp, y.row(i), R);
        if(is_sigmaknown==0){
          grad_sig += 1.0/S * calcu_grad_sig(old_sig, Theta_is, tid_wis);
        }
      }
    }
    
    // ---- calculate the final gradient and update params ----
    beta1t *= beta1;
    beta2t *= beta2;
    
    new_A   = update_param(old_A,   grad_A,   vu_A,   r_A,   beta1, beta2, beta1t, beta2t, eps, eta_A);
    new_gam = update_param(old_gam, grad_gam, vu_gam, r_gam, beta1, beta2, beta1t, beta2t, eps, eta_gam);
    new_B   = gamma2b(new_gam, R);
    
    if(is_sigmaknown==0){
      new_sig = update_param(old_sig, grad_sig, vu_sig, r_sig, beta1, beta2, beta1t, beta2t, eps, eta_sig);
      arma::vec sq_diag_sig = sqrt(new_sig.diag());
      new_sig.each_col() /= sq_diag_sig;
      new_sig.each_row() /= sq_diag_sig.t();
    }
    else{
      new_sig = old_sig;
    }
    
    // ---- stopping criterion ----
    new_B_nona = new_B;
    old_B_nona = old_B;
    
    new_B_nona.replace(NA_REAL,0);
    old_B_nona.replace(NA_REAL,0);
    error_para(0) = arma::norm(new_A      - old_A,      "fro")/arma::norm(old_A,      "fro"); // err A
    error_para(1) = arma::norm(new_B_nona - old_B_nona, "fro")/arma::norm(old_B_nona, "fro"); // err B
    error_para(2) = arma::norm(new_sig    - old_sig,    "fro")/arma::norm(old_sig,    "fro"); // err sig
    
    if(error_para.max() < tol_para){break;}
    
    // ---- replace params ----
    old_A = new_A;
    old_B = new_B;
    old_gam = new_gam;
    old_sig = new_sig;
    
  }
  
  double n2lb = calcu_n2lb(y, new_A, new_B, new_sig, mu_n, sigma_n, Fjr, Fjrp, N, K, S, M, R);
  
  List output = List::create(Rcpp::Named("new_A")   = new_A,
                             Rcpp::Named("new_B")   = new_B,
                             Rcpp::Named("new_sig") = new_sig,
                             Rcpp::Named("n2lb")    = n2lb,
                             Rcpp::Named("t")       = t,
                             
                             Rcpp::Named("vu_A")   = vu_A,
                             Rcpp::Named("r_A")    = r_A,
                             Rcpp::Named("vu_gam") = vu_gam,
                             Rcpp::Named("r_gam")  = r_gam,
                             Rcpp::Named("vu_sig") = vu_sig,
                             Rcpp::Named("r_sig")  = r_sig,
                             
                             Rcpp::Named("error_para") = error_para
                               
  );
  return(output);
  
}
