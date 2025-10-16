#define RCPP_ARMADILLO_RETURN_ANYVEC_AS_VECTOR
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;


void calcu_eta(arma::mat &eta,
               arma::mat ksi,
               const arma::Mat<int> &y,
               const arma::Mat<int> &z,
               const arma::Col<int> &R,
               const int ind){
  
  // Eyx3107, 2024.04.30
  // Calculate eta^{1}(ksi_{ij}) for index 1 and eta^{2}(ksi_{ij}) for index 2.
  // -- Input --
  // ksi: N*J mat, variational parameter.
  // y  : N*J mat, item graded response.
  // R  : J*1 vec, number of graded for each item.
  // ind: index, only 1 or 2.
  // -- Output --
  // eta: N*J mat.
  // -- Note --
  // 1. eta^{1}(ksi_{ij}) = - lambda(ksi_{ij,y_{ij}})
  //    eta^{2}(ksi_{ij}) = - lambda(ksi_{ij,y_{ij}+1})
  // 2. For computational simplicity, set lambda(ksi_{ij,y_{ij}}) = 0 for y_{ij} = 0
  //    and set lambda(ksi_{ij,y_{ij}+1}) = 0 for y_{ij} + 1 = R_{j}.
  // 3. For computational simplicity, set ksi_{ij,y_{ij}} = 1 for y_{ij} = 0
  //    and set ksi_{ij,y_{ij}+1} = 1 for y_{ij} + 1 = R_{j}.
  //    So, the first line ensure no error occur.
  
  ksi.elem( find(arma::abs(ksi) < 0.01) ).fill(0.01);
  eta = 0.5 / ksi % (0.5 - 1 / (1 + exp(ksi)));
  
  if(ind == 1){
    eta(find(y==0)).fill(0.0);
  }
  if(ind == 2){
    eta(find(z==0)).fill(0.0);
  }
  
}


arma::mat update_sigma_i(const arma::mat &omega,
                         const arma::mat &A,
                         const arma::rowvec &eta1_i,
                         const arma::rowvec &eta2_i){
  
  // Update the ith sigma in variational E-step. 2024.04.30.
  // omega: K*K mat, the inverse of current sigma.
  // A    : J*K mat, the current A.
  // eta1_i: J*1 vec, corresponding the ith row of current eta1.
  // eta2_i: J*1 vec, corresponding the ith row of current eta2.
  // -- Note --
  // eta^{1}_{ij} = 0 for y_{ij}     = 0
  // eta^{2}_{ij} = 0 for y_{ij} + 1 = R_{j}
  
  return(arma::inv(omega + 2 * A.t() * (A.each_col() % arma::trans(eta1_i + eta2_i))));
}


arma::rowvec update_mu_i(const arma::mat    &sigma_i,
                         const arma::rowvec &eta1_i,
                         const arma::rowvec &eta2_i,
                         const arma::mat &A,
                         const arma::mat &B,
                         const arma::Col<int> &R,
                         const arma::Row<int> &y_i){
  
  // Update the ith mu in variational E-step. 2024.04.30.
  // sigma_i: K*K mat, the ith sigma.
  // eta1_i : J*1 vec, correspoding the ith row of current eta1.
  // eta2_i : J*1 vec, correspoding the ith row of current eta2.
  // A      : J*K mat, the current A.
  // B      : J*max_R, the current B.
  // R      : J*1 vec, number of graded for each item.
  // y_i    : J*1 vec, item graded response for ith examinee.
  // -- Note --
  // eta^{1}_{ij} = 0 for y_{ij}     = 0
  // eta^{2}_{ij} = 0 for y_{ij} + 1 = R_{j}
  
  int J = A.n_rows;
  arma::rowvec beta1(J);
  arma::rowvec beta2(J);
  
  for(int j=0;j<J;j++){
    beta1(j) = (y_i(j)==0)     ?(-0.5):(2*B(j,y_i(j)-1)*eta1_i(j));
    beta2(j) = (y_i(j)==R(j)-1)? (0.5):(2*B(j,y_i(j))  *eta2_i(j));
  }
  
  return((beta1 + beta2)*A*sigma_i);
  
}


void ve_step(arma::cube &sigma_,       // dim(K,K,N)
             arma::mat  &mu_,          // dim(N,K)
             arma::cube &sig_mu_sum,   // dim(K,K,N)
             const arma::Mat<int> &y,
             const arma::mat  &old_A,
             const arma::mat  &old_B,
             const arma::Col<int> &R,
             const arma::mat  &old_sig,
             const arma::mat  &old_eta1,
             const arma::mat  &old_eta2
){
  
  // variational E-step for updating sigma_i, mu_i and sigma_mu_sum.
  
  int N = y.n_rows;
  int K = old_A.n_cols;
  arma::mat    omega = arma::inv(old_sig);
  arma::mat    sig_i_hat(K,K);
  arma::rowvec mu_i_hat(K);
  
  for(int i=0;i<N;i++){
    sig_i_hat = update_sigma_i(omega, old_A, old_eta1.row(i), old_eta2.row(i));
    mu_i_hat  = update_mu_i(sig_i_hat, old_eta1.row(i), old_eta2.row(i),
                            old_A, old_B, R, y.row(i));
    
    sigma_.slice(i) = sig_i_hat;
    mu_.row(i) = mu_i_hat;
    sig_mu_sum.slice(i) = sig_i_hat + mu_i_hat.t()*mu_i_hat;
  }
  
}


double obj_func_cpp(arma::mat sigma, arma::mat sigma_hat){
  arma::mat sigma_inv = arma::inv(sigma);
  return arma::accu( sigma_inv % sigma_hat ) + log(arma::det(sigma));
}


arma::mat calcu_sigma_cmle_cpp(arma::mat sigma_hat, arma::mat sigma0, double tol){
  
  arma::mat sigma1 = sigma0;
  arma::mat tmp = sigma0;
  double eps = 1;
  double step = 1;
  while(eps > tol){
    step = 1;
    tmp = arma::inv(sigma0);
    sigma1 = sigma0 - step * ( - tmp * sigma_hat * tmp + tmp );
    sigma1.diag().ones();
    sigma1 = arma::symmatu(sigma1);   // add 2021.04.25
    while(obj_func_cpp(sigma0, sigma_hat) < obj_func_cpp(sigma1, sigma_hat) ||
          min(arma::eig_sym(sigma1)) < 0){
      step *= 0.5;
      sigma1 = sigma0 - step * ( - tmp * sigma_hat * tmp + tmp );
      sigma1.diag().ones();
      sigma1 = arma::symmatu(sigma1);   // add 2021.04.25
    }
    eps = obj_func_cpp(sigma0, sigma_hat) - obj_func_cpp(sigma1, sigma_hat);
    sigma0 = sigma1;
  }
  return sigma0;
}


double update_ksi_ij(const arma::rowvec &Aj,
                     const arma::rowvec &Bj,
                     const arma::rowvec &mu_i,
                     const arma::mat &sigma_mu_i,
                     const int yij,
                     const int Rj,
                     const int ind,
                     const bool sq_rt = true){
  
  // Eyx3107, 2024.05.01
  // Calculate ksi^{1}_{ij} for index 1 and ksi^{2}_{ij} for index 2.
  // -- Input --
  // Aj        : K*1      vec, the jth row of A.
  // Bj        : max_Rj*1 vec, the jth row of B, length of Bj maybe larger than Rj.
  // mu_i      : K*1      vec, the ith row of mu_.
  // sigma_mu_i: K*K      mat, the ith page of sigma_mu_sum.
  // yij       : an integer  , item graded response for subject i to item j.
  // Rj        : an integer  , number of graded of item j.
  // ind       : index       , only 1 or 2.
  // sq_rt     : whether sqrt, default is TRUE.
  // -- Output --
  // ksi       : a numeric, variational parameter.
  // -- Note --
  // calculate ksi^{1}_{ij} = ksi_{ij,y_{ij}}   if ind = 1,
  // calculate ksi^{2}_{ij} = ksi_{ij,y_{ij}+1} if ind = 2.
  // For computational simplicity, 
  //   set ksi_{ij,y_{ij}}   = 1 for y_{ij} = 0,
  //   set ksi_{ij,y_{ij}+1} = 1 for y_{ij} + 1 = R_{j}.
  
  if(ind == 1 && yij == 0)      {return(1.0);}
  if(ind == 2 && yij == Rj - 1) {return(1.0);}
  
  double bj  = (ind == 1)?(Bj(yij-1)):(Bj(yij));
  double ksi_sq = pow(bj,2) - 2*bj*(Aj*mu_i.t()).eval()(0,0) + (Aj*sigma_mu_i*Aj.t()).eval()(0,0);
  
  if(sq_rt) {return(sqrt(ksi_sq));} else {return(ksi_sq);}
  
  // note that: could write as update ksi_j for all i.
}


arma::rowvec update_Aj(const arma::Row<int> &Mj,
                       const arma::rowvec &Bj,
                       const int Rj,
                       const arma::Col<int> &yj,
                       const arma::cube &sig_mu_sum,
                       const arma::mat &mu_,
                       const arma::vec &eta1_j,
                       const arma::vec &eta2_j){
  
  // Eyx3107, 2024.05.01
  // Update A for certain j.
  // -- Input --
  // Bj        : max_Rj*1 vec, the jth row of B, length of Bj maybe larger than Rj.
  // Rj        : an   integer, number of graded of item j.
  // y_j       : N*1      vec, item graded response to item j for all i.
  // sig_mu_sum: N*K*K  array, sigma + mu%*%t(mu) for all i.
  // mu_       : N*K      mat, mu for all i and k.
  // eta1_j    : N*1      vec, eta^{1}_{ij} for certain j and all i.
  // eta2_j    : N*1      vec, eta^{2}_{ij} for certain j and all i.
  // ind       : a     vector, index for non-zero elements of Aj.
  // -- Output --
  // Aj: K*1 vec. 
  //  
  // -- Note --
  // eta^{1}_{ij} = 0 for y_{ij}     = 0
  // eta^{2}_{ij} = 0 for y_{ij} + 1 = R_{j}
  // beta^{1}_{ij} = -.5 for y_{ij} = 0
  // beta^{1}_{ij} = eta^{1}_{ij}*b_{j,y_{ij}} for 1 <= y_{ij} <= R_{j}-1.
  // beta^{2}_{ij} = -.5 for y_{ij} + 1 = R_{j}
  // beta^{2}_{ij} = eta^{2}_{ij}*b_{j,y_{ij}+1} for 1 <= y_{ij} + 1 <= R_{j}-1.

  int K = Mj.n_elem;
  int k = sum(Mj);
  
  arma::rowvec Aj = arma::zeros<arma::rowvec>(K);
  
  if(k == 0){
    return(Aj);
  }
  
  arma::mat sig_mu_sum_sum = arma::zeros(K,K);
  arma::uvec ind = find(Mj==1);
  arma::mat    first_term(k,k);
  arma::rowvec second_term(k);
  
  // -- first term --
  int i;
  int N = yj.n_elem;
  for(i=0;i<N;i++){
    sig_mu_sum_sum += (eta1_j(i) + eta2_j(i)) * sig_mu_sum.slice(i);
  }
  
  first_term = 0.5 * arma::inv(sig_mu_sum_sum(ind, ind));
  
  // -- second term --
  arma::rowvec beta1(N);
  arma::rowvec beta2(N);
  
  for(i=0;i<N;i++){
    beta1(i) = (yj(i)==0   )?(-0.5):(2*Bj(yj(i)-1)*eta1_j(i));
    beta2(i) = (yj(i)==Rj-1)?( 0.5):(2*Bj(yj(i)  )*eta2_j(i));
  }
  
  second_term = (beta1 + beta2)*mu_.cols(ind);
  Aj(ind) = second_term * first_term;
  
  return(Aj);
  
}


arma::rowvec update_Bj(const arma::rowvec &Aj,
                       const arma::rowvec &Bj,
                       const int Rj,
                       const arma::mat &mu_,
                       const arma::vec &eta1_j,
                       const arma::vec &eta2_j,
                       const arma::Col<int> &yj){
  
  // Eyx3107, 2024.05.03
  // Update B for certain j.
  // -- Input --
  // Aj    : K*1      vec, the jth row of A.
  // Bj    : max_Rj*1 vec, the jth row of B, length of Bj maybe larger than Rj.
  // Rj    : an   integer, number of graded of item j.
  // mu_   : N*K      mat, mu for all i and k.
  // eta1_j: N*1      vec, eta^{1}_{ij} for certain j and all i.
  // eta2_j: N*1      vec, eta^{2}_{ij} for certain j and all i.
  // yj    : N*1      vec, item graded response to item j for all i.
  // -- Output --
  // Bj: max_Rj*1 vec.
  //  
  // -- Note --
  // eta^{1}_{ij} = 0 for y_{ij}     = 0
  // eta^{2}_{ij} = 0 for y_{ij} + 1 = R_{j}
  // For r > 2:
  // beta^{1}_{ij} = .0 for y_{ij} <= r - 2
  // beta^{1}_{ij} = .5 for y_{ij}  = r - 1
  // beta^{1}_{ij} = 2*eta^{1}_{ij}*(b_{j,y_{ij}} - gam_{j,r}) for r <= y_{ij} <= R_{j}-1.
  // beta^{2}_{ij} = .0 for y_{ij} <= r - 2
  // beta^{2}_{ij} = .5 for y_{ij} = R_{j} - 1
  // beta^{2}_{ij} = 2*eta^{2}_{ij}*(b_{j,y_{ij}+1} - gam_{j,r}) for r-1 <= y_{ij} <= R_{j}-2.
  
  // -- re-params --
  arma::rowvec gamj = Bj;
  if(Rj > 2){
    gamj.subvec(1,Rj-2) = arma::diff(Bj.subvec(0,Rj-2));
  }
  
  // -- r = 1 --
  double denom  = 2*sum(eta1_j + eta2_j);
  double numer1 = 2*((eta1_j + eta2_j).t()*mu_*Aj.t()).eval()(0,0);
  
  int i;
  int N = yj.n_elem;
  arma::vec beta1(N);
  arma::vec beta2(N);
  for(i=0;i<N;i++){
    beta1(i) = (yj(i)==0)   ?(-0.5):(2*(Bj(yj(i)-1) - Bj(0))*eta1_j(i));
    beta2(i) = (yj(i)==Rj-1)?( 0.5):(2*(Bj(yj(i))   - Bj(0))*eta2_j(i));
  }
  double numer2 = - sum(beta1 + beta2);
  
  gamj(0) = (numer1 + numer2)/denom;
  
  // -- r > 1 --
  if(Rj > 2){
    
    int r;
    arma::vec eta1_j__(N);
    arma::vec eta2_j__(N);
    arma::vec beta1(N);
    arma::vec beta2(N);
    
    double para_a;
    double para_b, para_b1, para_b2, para_bp; // b_prime
    int    para_c;
    double func_h;
    
    for(r=2;r<=Rj-1;r++){
      
      eta1_j__ = eta1_j;
      eta2_j__ = eta2_j;
      eta1_j__.elem(find(yj<r  )).zeros();
      eta2_j__.elem(find(yj<r-1)).zeros();
      
      arma::uvec ind = find(yj == r-1);
      para_c = ind.n_elem;
      para_a = -2*sum(eta1_j__ + eta2_j__);
      para_b1 = 2*((eta1_j__ + eta2_j__).t()*mu_*Aj.t()).eval()(0,0);
      
      beta1.zeros();
      beta2.zeros();
      beta1.elem(find(yj == r -1)).fill(0.5);
      beta2.elem(find(yj == Rj-1)).fill(0.5);
      
      for(i=0;i<N;i++){
        beta1(i) = (yj(i)<r)                   ?(beta1(i)):(2*eta1_j(i)*(Bj(yj(i)-1)-gamj(r-1)));
        beta2(i) = ((yj(i)==Rj-1)|(yj(i)<=r-2))?(beta2(i)):(2*eta2_j(i)*(Bj(yj(i)  )-gamj(r-1)));
      }
      
      para_b2 = - sum(beta1 + beta2);
      para_b  = para_b1 + para_b2;
      
      func_h  = exp(-gamj(r-1)) + gamj(r-1) - 1;
      para_bp = para_b + para_a * func_h;
      gamj(r-1) = 0.5/para_a * (- para_bp - sqrt(pow(para_bp,2) - 4*para_a*para_c)) + func_h;
      
    }
    
  }
  
  // -- recovery params --
  arma::rowvec new_Bj = gamj;
  if(Rj > 2){
    new_Bj.subvec(0,Rj-2) = arma::cumsum(gamj.subvec(0,Rj-2));
  }
  
  return(new_Bj);

}


double calcu_n2vlb_j(const arma::rowvec &Aj,
                     const arma::rowvec &Bj,
                     const arma::mat &mu_,
                     const arma::cube &sig_mu_sum,
                     const arma::Col<int> &yj,
                     const int Rj,
                     const arma::vec &ksi1_j,
                     const arma::vec &ksi2_j,
                     const arma::vec &eta1_j,
                     const arma::vec &eta2_j){
  
  // Eyx3107, 2024.05.14
  // Compute -2 variational lower bound for For certain j.
  // -- Input --
  // Aj        : K*1      vec, the jth row of A.
  // Bj        : max_Rj*1 vec, the jth row of B, length of Bj maybe larger than Rj.
  // mu_       : N*K      mat, mu for all i and k.
  // sig_mu_sum: N*K*K  array, sigma + mu%*%t(mu) for all i.
  // Yj        : N*1      vec, item graded response to item j for all i.
  // Rj        : an   integer, number of graded of item j.
  // ksi1_j    : N*1      vec, variational param, ksi^{1}_{ij} for certain j and all i.
  // ksi2_j    : N*1      vec, variational param, ksi^{2}_{ij} for certain j and all i.
  // eta1_j    : N*1      vec, eta^{1}_{ij} for certain j and all i.
  // eta2_j    : N*1      vec, eta^{2}_{ij} for certain j and all i.
  // cri       : loglik, aic, bic or ebic.
  // ga        : parameter for ebic.
  //   
  // -- Output --
  // n2vlb_j: a numeric, -2 variational lower bound for certain j.
  //   
  // -- Note --
  // ksi^{1}_{ij} = 1 for y_{ij}     = 0
  // ksi^{2}_{ij} = 0 for y_{ij} + 1 = R_{j}
  // eta^{1}_{ij} = 0 for y_{ij}     = 0
  // eta^{2}_{ij} = 0 for y_{ij} + 1 = R_{j}
  //   
  // The calculation of -.5*\tilde{U}_{j} has 3 terms need to be computed:
  // term 1:
  //   \sum_{i=1}^{N} { [log(F(ksi^{1}_{ij}))   - .5*ksi^{1}_{ij}] * (Y_{ij}!=0) }
  //  +\sum_{i=1}^{N} { [log(1-F(ksi^{2}_{ij})) + .5*ksi^{2}_{ij}] * (Y_{ij}!=Rj-1) }
  // term 2:
  //   \sum_{i=1}^{N} { -.5*Bj[Y_{ij}]   }  # Bj_{0}  = 0
  //  +\sum_{i=1}^{N} {  .5*Bj[Y_{ij}+1] }  # Bj_{Rj} = 0
  //  +\sum_{i=1}^{N} { I(Y_{ij}!=0 & Y_{ij}!=Rj-1) * log(1 - exp(Bj[Y_{ij}]-Bj[Y_{ij}+1])) }
  //  +\sum_{i=1}^{N} { I(Y_{ij}=Rj-1)*.5*t(Aj)%*%mu_{i} - I(Y_{ij}=0)*.5*t(Aj)%*%mu_{i}    }
  // term 3:    
  //   \sum_{i=1}^{N} { - eta^{1}_{ij}*[ update_ksi()^2 - (ksi^{1}_{ij})^2 ] }
  //  +\sum_{i=1}^{N} { - eta^{2}_{ij}*[ update_ksi()^2 - (ksi^{2}_{ij})^2 ] }
  
  int N = yj.n_elem;
  
  double term1;
  double term2;
  double term3, term3_1_i, term3_2_i;
  double loglik;
  
  // -- term1 --
  arma::vec term1_1 = 2*log(1 + exp(-ksi1_j)) + ksi1_j;
  arma::vec term1_2 = 2*log(1 + exp( ksi2_j)) - ksi2_j;
  
  term1_1.elem(find(yj==0   )).zeros();
  term1_2.elem(find(yj==Rj-1)).zeros();
  term1 = sum(term1_1) + sum(term1_2);
  
  // -- term2 --
  arma::vec b1(N);
  arma::vec b2(N);
  arma::vec beta = mu_*Aj.t();
  beta.elem(find(yj==Rj-1)) *= - 1;
  
  int i;
  for(i=0;i<N;i++){
    b1(i) = (yj(i)==0   )?(0):(Bj(yj(i)-1));
    b2(i) = (yj(i)==Rj-1)?(0):(Bj(yj(i)  ));
    beta(i) = ((yj(i)==0)|(yj(i)==Rj-1))?(beta(i)):(-2*log(1-exp(Bj(yj(i)-1)-Bj(yj(i)))));
  }
  term2 = sum(b1) - sum(b2) + sum(beta);
  
  // -- term3 --
  term3 = 0.0;
  for(i=0;i<N;i++){
    term3_1_i = eta1_j(i)*(update_ksi_ij(Aj, Bj, mu_.row(i), sig_mu_sum.slice(i), yj(i), Rj, 1, false) - pow(ksi1_j(i),2));
    term3_2_i = eta2_j(i)*(update_ksi_ij(Aj, Bj, mu_.row(i), sig_mu_sum.slice(i), yj(i), Rj, 2, false) - pow(ksi2_j(i),2));
    term3 += 2*(term3_1_i + term3_2_i);
  }
  
  // -- loglik --
  loglik = term1 + term2 + term3;
  return(loglik);
  
}


double calcu_n2vlb_0(const arma::mat &sig,
                     const arma::mat &hat_sig,
                     const int N){
  
  return(N*arma::log_det_sympd(sig) + N*arma::accu(arma::inv(sig)%hat_sig));
}
  
  
double calcu_entro(const arma::cube &sigma_,
                   const int N,
                   const int K){
  int i;
  double entro = 0.0;
  for(i=0;i<N;i++){
    entro -= arma::log_det_sympd(sigma_.slice(i));
  }
  entro -= N*K;
  return(entro);
}


// [[Rcpp::export]]
Rcpp::List rcpp_gvemgrm(
    const arma::Mat<int> &y,
    const arma::Col<int> &R,
    arma::mat old_A,
    arma::mat old_B,
    arma::mat old_sig,
    arma::mat old_ksi1,
    arma::mat old_ksi2,
    
    const arma::Mat<int> Mod,
    const int    is_sigmaknown,
    const int    max_iter = 100,
    const double tol_n2vlb = 1e-4,
    const double tol_para = 1e-3,
    const int    stop_cri = 2,
    const int    is_calcu_n2vlb = 0

){
  
  // y:        N*J    mat, observed item responses.
  // R:        J*1    vec, number of graded score for each item.
  // old_A:    J*K    mat, initial value of A.
  // old_B:    J*maxR vec, initial value of B.
  // old_sig:  K*K    mat, initial value of \Sigma.
  // old_ksi1: N*J    mat, initial value of \ksi^{1}.
  // old_ksi2: N*J    mat, initial value of \ksi^{2}.
  // Mod:      J*K    mat, target model (structure of A), the element only 0 or 1 is valid.
  // is_sigmaknown: 0 denotes sigma unknown, 1 denotes sigma known.
  // max_iter: number maximum of iterations, defualt is 100.
  // tol_n2vlb: tolerance for variational lower bound, default is 1e-4.
  // tol_para:  tolerance for parameters, default is 1e-3.
  // stop_cri: stopping criterion, 1 denotes n2vlb, 2 denotes params
  // is_calcu_n2vlb: whether calculate n2vlb in each iteration when stop_cri = 2, default is 0.
  
  int N = y.n_rows;
  int J = y.n_cols;
  int K = old_A.n_cols;
  
  arma::mat old_eta1(N,J);
  arma::mat old_eta2(N,J);
  double old_n2vlb = 1.0;
  
  arma::Mat<int> z = R.t() - y.each_row() - 1;  // reverse of y.
  calcu_eta(old_eta1, old_ksi1, y, z, R, 1);
  calcu_eta(old_eta2, old_ksi2, y, z, R, 2);
  
  arma::mat new_A     = old_A;
  arma::mat new_B     = old_B;
  arma::mat new_sig   = old_sig;
  arma::mat new_ksi1  = old_ksi1;
  arma::mat new_ksi2  = old_ksi2;
  arma::mat new_eta1  = old_eta1;
  arma::mat new_eta2  = old_eta2;
  double    new_n2vlb = old_n2vlb;
  
  // ---- work array ----
  arma::cube sigma_n(K,K,N);
  arma::mat  mu_n(N,K);
  arma::cube sig_mu_sum_n(K,K,N);
  arma::mat  hat_sig(K,K);
  arma::vec  n2vlb_vec(J);
  double     n2vlb_0;
  double     entro;
  double     error_n2vlb;
  arma::vec  error_para(3);
  int i,j;
  
  arma::mat old_B_nona = old_B; // replace na with 0 for calcu err B
  arma::mat new_B_nona = new_B; // replace na with 0 for calcu err B
  
  int converge_para  = 0;
  int converge_n2vlb = 0;
  
  // ---- vem iteration ----
  arma::vec n2vlb_seq(max_iter);
  int it = 0;
  while(it < max_iter){
    
    it += 1;
    Rprintf("it: %03d\r", it);
    
    // -- ve-step: update sigma_n, mu_n, sig_mu_sum_n --
    ve_step(sigma_n, mu_n, sig_mu_sum_n, y, old_A, old_B, R, old_sig, old_eta1, old_eta2);
    
    // -- m-step --
    // -- 1. update sigma_theta --
    if(is_sigmaknown == 0){
      hat_sig = sum(sig_mu_sum_n, 2)/N;
      new_sig = calcu_sigma_cmle_cpp(hat_sig, old_sig, 1e-6);
    }
    
    // -- 2.1 update ksi and eta --
    for(j=0;j<J;j++){
      for(i=0;i<N;i++){
        new_ksi1(i,j) = update_ksi_ij(old_A.row(j), old_B.row(j), mu_n.row(i), sig_mu_sum_n.slice(i), y(i,j), R(j), 1);
        new_ksi2(i,j) = update_ksi_ij(old_A.row(j), old_B.row(j), mu_n.row(i), sig_mu_sum_n.slice(i), y(i,j), R(j), 2);
      }
    }
    
    calcu_eta(new_eta1, new_ksi1, y, z, R, 1);
    calcu_eta(new_eta2, new_ksi2, y, z, R, 2);
    
    // -- 2.2 update B --
    for(j=0;j<J;j++){
      new_B.row(j) = update_Bj(old_A.row(j), old_B.row(j), R(j), mu_n,
                new_eta1.col(j), new_eta2.col(j), y.col(j));
    }
    
    // -- 2.3 update A --
    for(j=0;j<J;j++){
      new_A.row(j) = update_Aj(Mod.row(j), new_B.row(j), R(j), y.col(j),
                sig_mu_sum_n, mu_n, new_eta1.col(j), new_eta2.col(j));
    }
    
    // -- 2.4 calculate n2vlb --
    if(stop_cri == 1 || is_calcu_n2vlb == 1){
      for(j=0;j<J;j++){
        n2vlb_vec(j) = calcu_n2vlb_j(new_A.row(j), new_B.row(j), mu_n, sig_mu_sum_n,
                  y.col(j), R(j), new_ksi1.col(j), new_ksi2.col(j),
                  new_eta1.col(j), new_eta2.col(j));
      }
      
      n2vlb_0 = calcu_n2vlb_0(new_sig, hat_sig, N);
      entro   = calcu_entro(sigma_n, N, K);
      new_n2vlb = n2vlb_0 + sum(n2vlb_vec) + entro;
      n2vlb_seq(it-1) = new_n2vlb;
    }
    
    
    // -- stopping criterion --
    error_n2vlb = abs(old_n2vlb-new_n2vlb)/abs(old_n2vlb);
    error_para(0) = arma::norm(new_A - old_A, "fro")/arma::norm(old_A, "fro");       // err A
    new_B_nona = new_B;
    old_B_nona = old_B;
    new_B_nona.replace(NA_REAL,0);
    old_B_nona.replace(NA_REAL,0);
    error_para(1) = arma::norm(new_B_nona - old_B_nona, "fro")/arma::norm(old_B_nona, "fro");       // err B
    error_para(2) = arma::norm(new_sig - old_sig, "fro")/arma::norm(old_sig, "fro"); // err sig
    
    // -- check whether stop --
    converge_n2vlb = 0;
    converge_para  = 0;
    if(error_n2vlb < tol_n2vlb){
      converge_n2vlb = 1;
    }
    if(error_para.max() < tol_para){
      converge_para = 1;
    }
    
    if(stop_cri == 1){
      if(converge_n2vlb == 1){
        break;
      }
    }
    else{
      if(converge_para == 1){
        break;
      }
    }
    
    // -- replace params --
    old_A     = new_A;
    old_B     = new_B;
    old_sig   = new_sig;
    old_ksi1  = new_ksi1;
    old_ksi2  = new_ksi2;
    old_eta1  = new_eta1;
    old_eta2  = new_eta2;
    old_n2vlb = new_n2vlb;
    
  } // end while
  
  
  if(stop_cri == 2 && is_calcu_n2vlb==0){
    converge_n2vlb = NA_INTEGER;
    for(j=0;j<J;j++){
      n2vlb_vec(j) = calcu_n2vlb_j(new_A.row(j), new_B.row(j), mu_n, sig_mu_sum_n,
                y.col(j), R(j), new_ksi1.col(j), new_ksi2.col(j),
                new_eta1.col(j), new_eta2.col(j));
    }
    
    n2vlb_0 = calcu_n2vlb_0(new_sig, hat_sig, N);
    entro   = calcu_entro(sigma_n, N, K);
    new_n2vlb = n2vlb_0 + sum(n2vlb_vec) + entro;
  }
  
  List output = List::create(Rcpp::Named("new_A")    = new_A,
                             Rcpp::Named("new_B")    = new_B,
                             Rcpp::Named("new_sig")  = new_sig,
                             Rcpp::Named("new_ksi1") = new_ksi1,
                             Rcpp::Named("new_ksi2") = new_ksi2,
                             Rcpp::Named("mu_n")     = mu_n,
                             Rcpp::Named("sigma_n")  = sigma_n,
                             Rcpp::Named("n2vlb")    = new_n2vlb,
                             Rcpp::Named("n2vlb_seq")= n2vlb_seq.subvec(0,it-1),
                             Rcpp::Named("it")       = it,
                             Rcpp::Named("converge_n2vlb")  = converge_n2vlb,
                             Rcpp::Named("converge_para") = converge_para
                             
                             
  );
  return(output);
}
