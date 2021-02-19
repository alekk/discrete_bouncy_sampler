#include <iostream>
#include <armadillo>
#include <vector>


using namespace std;


// Null functions
// e.g. for flat priors
double l_null(const arma::vec &x, const std::vector<double> &thetas) {
  return 0.0;
}
arma::dvec gl_null(const arma::vec &x, const std::vector<double> &thetas) {
  return x*0;
}
double gldote_null(const arma::vec &x, const arma::vec &e, const std::vector<double> &thetas) {
  return 0.0;
}

// Gauss Iso
double l_GaussIso(const arma::vec &x, const std::vector<double> &thetas) {
  arma::vec x2=x%x;
  double sig2=thetas[0];
  return -0.5*arma::accu(x2)/sig2;
}
arma::dvec gl_GaussIso(const arma::vec &x, const std::vector<double> &thetas) {
  double sig2=thetas[0];
  return -x/sig2;
}
double gldote_GaussIso(const arma::vec &x, const arma::vec &e, const std::vector<double> &thetas) {
  double sig2=thetas[0];
  return -arma::dot(x,e)/sig2/arma::norm(e);
}
// Power iso log f = -||x||^a/a
double l_PowIso(const arma::vec &x, const std::vector<double> &thetas) {
  double norm2=arma::accu(x%x);
  double a=thetas[0];
  return -exp(a/2*log(norm2))/a;
}
arma::dvec gl_PowIso(const arma::vec &x, const std::vector<double> &thetas) {
  double norm2=arma::accu(x%x);
  double a=thetas[0];
  return -exp((a/2-1)*log(norm2))*x;
}
double gldote_PowIso(const arma::vec &x, const arma::vec &e, const std::vector<double> &thetas) {
  double norm2=arma::accu(x%x);
  double a=thetas[0];
  return -exp((a/2-1)*log(norm2))*arma::dot(x,e)/arma::norm(e);
}

// Gauss diag specifying each diagonal expectation and variance
// Expectation \ne 0 can be useful when using this as a prior
double l_GaussDiag(const arma::vec &x, const std::vector<double> &theta) {
  int d=x.n_elem, i;
  double ll=0;
  for (i=0;i<d;i++) {
    ll+=(x(i)-theta[i])*(x(i)-theta[i])/theta[i+d];
  }
  return -0.5*ll;
}

arma::dvec gl_GaussDiag(const arma::vec &x, const std::vector<double> &theta) {
  arma::dvec gl=x;
  int i, d=x.n_elem;
  for (i=0;i<d;i++) {
    gl(i)=(x(i)-theta[i])/theta[i+d];
  }
  return -gl;
}

// SDs linear from theta[0] to theta[1]
double l_GaussLinSD(const arma::vec &x, const std::vector<double> &thetas) {
  arma::vec sds=arma::linspace<arma::dvec>(thetas[0],thetas[1],x.n_elem);
  return -0.5*arma::accu((x%x)/(sds%sds));
}
arma::dvec gl_GaussLinSD(const arma::vec &x, const std::vector<double> &thetas) {
  arma::vec sds=arma::linspace<arma::dvec>(thetas[0],thetas[1],x.n_elem);
  return -x/(sds%sds);
}
double gldote_GaussLinSD(const arma::vec &x, const arma::vec &e, const std::vector<double> &thetas) {
  arma::vec sds=arma::linspace<arma::dvec>(thetas[0],thetas[1],x.n_elem);
  return -arma::dot(e,x/(sds%sds))/arma::norm(e);
}
// Power aniso log f = -||x/sdvec||^a/a
double l_PowLinSc(const arma::vec &x, const std::vector<double> &thetas) {
  arma::dvec sds=arma::linspace<arma::dvec>(thetas[1],thetas[2],x.n_elem);
  arma::dvec xstd=x/sds;
  double norm2=arma::accu(xstd%xstd);
  double a=thetas[0];
  return -exp(a/2*log(norm2))/a;
}
arma::dvec gl_PowLinSc(const arma::vec &x, const std::vector<double> &thetas) {
  arma::dvec sds=arma::linspace<arma::dvec>(thetas[1],thetas[2],x.n_elem);
  arma::dvec xstd=x/sds;
  double norm2=arma::accu(xstd%xstd);
  double a=thetas[0];
  return -exp((a/2-1)*log(norm2))*xstd/sds;
}
double gldote_PowLinSc(const arma::vec &x, const arma::vec &e, const std::vector<double> &thetas) {
  arma::dvec sds=arma::linspace<arma::dvec>(thetas[1],thetas[2],x.n_elem);
  arma::dvec xstd=x/sds;
  double norm2=arma::accu(xstd%xstd);
  double a=thetas[0];
  return -exp((a/2-1)*log(norm2))*arma::dot(xstd/sds,e)/arma::norm(e);
}


// Hessian linear from theta[0]/d to theta[0]
double l_GaussLinH(const arma::vec &x, const std::vector<double> &thetas) {
  int d=x.n_elem;
  double dd=(double)d;
  arma::vec Hesss=arma::linspace<arma::dvec>(thetas[0]/dd,thetas[0],d);
  return -0.5*arma::accu(x%x%Hesss);
}
arma::dvec gl_GaussLinH(const arma::vec &x, const std::vector<double> &thetas) {
  int d=x.n_elem;
  double dd=(double)d;
  arma::vec Hesss=arma::linspace<arma::dvec>(thetas[0]/dd,thetas[0],d);
  return -x%Hesss;
}
double gldote_GaussLinH(const arma::vec &x, const arma::vec &e, const std::vector<double> &thetas) {
  int d=x.n_elem;
  double dd=(double)d;
  arma::vec Hesss=arma::linspace<arma::dvec>(thetas[0]/dd,thetas[0],d);
  return -arma::dot(e,x%Hesss)/arma::norm(e);
}


// Student t
double l_Student_t_gen(const arma::vec &x, const std::vector<double> &thetas) {
  arma::vec x2=x%x;
  int d=x.n_elem, i,j, thelem=d+1;
  arma::vec mu(x);
  arma::mat SigInv(d,d);
  double nu=thetas[0];
  
  for (i=0;i<d;i++) {
    mu(i)=thetas[i+1];
    for (j=0;j<d;j++) {
      SigInv(i,j)=thetas[thelem++];
    }
  }
  mu=x-mu;
  arma::mat B=mu.t()*SigInv*mu;
  //cout << "Student t "<< mu.t() << ", "<<B<<", "<<B(0,0)<<"\n"<<SigInv<<"\n";
  return -(nu+d)/2*log(1+B(0,0)/nu);
}

// Bimodal Gaussian with SD=1 and modes at +/- theta(0),
// Prob of 1st (+) mode is theta(1).
double l_GaussBimod(const arma::vec &x, const std::vector<double> &thetas) {
  double mu1=thetas[0], mu2=-mu1, p1=thetas[1], p2=1-p1;
  arma::vec x12=(x-mu1)%(x-mu1);
  arma::vec x22=(x-mu2)%(x-mu2);
  double norm21=arma::accu(x12), norm22=arma::accu(x22);
  double a,b;
  if (norm21<norm22) {
    a=log(p1)-0.5*norm21;
    b=log(1+p2/p1*exp(0.5*(norm21-norm22)));
  }
  else {
    a=log(p2)-0.5*norm22;
    b=log(1+p1/p2*exp(0.5*(norm22-norm21)));
  }
    
  return a+b;
}
