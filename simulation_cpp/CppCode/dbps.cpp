#include <iostream>
#include <armadillo>
#include <vector>
#include <chrono>
#include <random>
#include "toy_targets.h"


// DBPS 

using namespace std;

static void InsertInOutput(arma::vec x,double lpost, arma::mat &Output, int &row) {
  int i, n=x.n_elem;
  for (i=0;i<n;i++) {
    Output(row,i)=x(i);
  }
  Output(row,i)=lpost;
  row++;
}
static void PrintInfo(int it, double lpri, double ll, const arma::vec &x,
		      const arma::vec &u, int nstd, int nbounce, int nrej,
		      double udotsum) {
  double di=(double)(it+1), db=(double)nbounce, drej=(double)nrej;
  cout << "**Iteration="<<it+1<<"\n";
  cout << "lprior="<<lpri<<", ll="<<ll<<"\n";
  cout << "x="<<x.t();
  cout << "u="<<u.t();
  cout << "alphastd="<<(double)nstd/di<<", alphadr="<< db/(db+drej)<< ", frac succ bou="<<db/di<<"\n";
  cout << "udot mean ="<< udotsum/(db+drej)<<"\n";
}




// Reflect velocity in gradient
arma::vec reflect(const arma::vec &uc, const arma::vec &g) {
  arma::vec gn=normalise(g);
  return uc-2.0*dot(gn,uc)*gn;
}

// Reflect velocity in gradient using preconditioning
// true co-ord = A * isotropic co-ord
arma::vec reflectA(const arma::vec &uc, const arma::vec &g, const arma::mat &A) {
  arma::mat Sigma=A*A.t();
  arma::mat QF=g.t()*Sigma*g;
  double denom=QF(0,0);
  return uc-2.0*dot(g,uc)*Sigma*g/denom;
}

// Refresh velocity by adding on a Gaussian and renormalising
void refresh_u(double beta, arma::dvec &u) {
  double norman=arma::norm(u);
  arma::dvec xi=arma::dvec(size(u));
  xi.randn(); xi=xi/sqrt((double)u.n_elem);
  u=norman*arma::normalise(beta*u+sqrt(1-beta*beta)*xi);
}
// Precon: B B^T=A
void refresh_uA(double beta, arma::dvec &u, arma::mat A) {
  arma::vec util=solve(A,u);
  double norman=arma::norm(util);
  arma::dvec xi=arma::dvec(size(u));
  xi.randn(); xi=xi/sqrt((double)u.n_elem);
  util=norman*arma::normalise(beta*util+sqrt(1-beta*beta)*xi);
  u=A*util;
}

//**********************************************************
// Arguments to dbps
// nits = number of iterations
// x0, u0 = initial position and velocity
// delta, kappa = tuning parameters
// Sigma = preconditioning matrix
// pll_fn = pointer to log-likelihood function
// plpri_fn = pointer to log-prior function
// rootname = root for output file names
// thin = thinning factor for use before storing and outputing
// prt = print output every k iterations (or never)

void dbps(int nits, const arma::vec &x0, const arma::vec &u0,
	  const vector<double>thetall, const vector<double>thetapri,
	  double delta, double kappa, const arma::mat Sigma,
	  double (*pll_fn)(const arma::vec &xs, const vector<double> &theta),
	  arma::vec (*pgll_fn)(const arma::vec &xs, const vector<double> &theta),
	  double (*plpri_fn)(const arma::vec &xs, const vector<double> &theta),
	  arma::vec (*pglpri_fn)(const arma::vec &xs, const vector<double> &theta),
	  string rootname="",
	  int thin=1, int prt=0) {
  int d=x0.n_elem;
  int nout = 1+nits/thin, outrow=0;
  double beta=exp(-kappa*delta/2);
  arma::dmat Output(nout,d+1);
  double lpricurr=(*plpri_fn)(x0,thetapri), lpriprop, lpripropprop;
  double llcurr=(*pll_fn)(x0,thetall), llprop, llpropprop;
  arma::vec xcurr=x0, xprop=arma::vec(size(x0)), xpropprop=arma::vec(size(x0));
  arma::vec ucurr=u0, uref=arma::vec(size(x0));
  double lalpha1,lalpha2,lalphadr;
  int i, nacc1=0, nrej=0, nbounce=0;
  bool precon=(arma::accu(abs(Sigma-arma::eye(size(Sigma))))>0);
  arma::mat SigInv=arma::inv_sympd(Sigma);
  arma::mat A=arma::sqrtmat_sympd(Sigma).t();
  string outfnameroot="dbps"+rootname+ "del" + to_string((int) (delta*1000))+"kappa"+to_string((int)(kappa*1000)) + "A"+to_string((int) precon);
  string outfname=outfnameroot+".txt";
  string outmetaname=outfnameroot+".info";
  double udotsum=0;
  arma::vec ujustpost=ucurr;
  
  std::random_device rd;  //Will use to obtain seed for random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> Unif(0.0, 1.0);

  InsertInOutput(xcurr,llcurr+lpricurr,Output, outrow);

  auto t1 = std::chrono::high_resolution_clock::now();

  for (i=0;i<nits;i++) {
    if (((i+1) % prt == 0) && (i!=0)) {
      PrintInfo(i,lpricurr,llcurr,xcurr,ucurr,nacc1,nbounce,nrej,udotsum);
    }

    xprop=xcurr+delta*ucurr;
    llprop=(*pll_fn)(xprop,thetall);
    lpriprop=(*plpri_fn)(xprop,thetapri);
    lalpha1=lpriprop+llprop-lpricurr-llcurr;

    if (log(Unif(gen))<lalpha1) { // standard movement
      xcurr=xprop; llcurr=llprop; lpricurr=lpriprop;
      nacc1++;
    }
    else { // try delayed rejection
      udotsum += arma::dot(ucurr,SigInv*ujustpost);
      arma::dvec g=(*pglpri_fn)(xprop,thetapri)+(*pgll_fn)(xprop,thetall);
      if (precon) {
	uref=reflectA(ucurr, g, A);
      }
      else {
	uref=reflect(ucurr, g);
      }
      xpropprop=xprop+delta*uref;
      llpropprop=(*pll_fn)(xpropprop,thetall);
      lpripropprop=(*plpri_fn)(xpropprop,thetapri);
      lalpha2=lpriprop+llprop-lpripropprop-llpropprop;
      lalpha2=(lalpha2<0)?lalpha2:0;
      lalpha1=(lalpha1<0)?lalpha1:0;
      //      cout << "stuff:a1="<<lalpha1<<", a2="<<lalpha2<<",\n ";
      lalphadr=llpropprop+lpripropprop-llcurr-lpricurr;
      lalphadr+=log(1-exp(lalpha2))-log(1-exp(lalpha1));
      if (log(Unif(gen))<lalphadr) { // dr successful
	xcurr=xpropprop; llcurr=llpropprop; lpricurr=lpripropprop;
	ucurr=uref;
	nbounce++;
      }
      else { // dr unsuccessful
	ucurr=-ucurr;
	nrej++;
      }
      ujustpost=ucurr;
    }

    // velocity refresh
    if (precon) {
      refresh_uA(beta,ucurr,A);
    }
    else {
      refresh_u(beta,ucurr);
    }
    
    if ((i+1) % thin == 0) {
      InsertInOutput(xcurr,llcurr+lpricurr,Output,outrow);
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();

  Output.save(string("./")+outfname,arma::raw_ascii);
  double dnits=(double)nits, dnbounce=(double)nbounce, dnrej=(double)nrej;
  double dnacc1=(double)nacc1;
  ofstream meta;
  meta.open(string("./"+outmetaname));
  meta <<"\n***";
  meta << "nits=" <<nits<<", delta="<<delta<<", kappa="<<kappa<<", precon="<<precon<<"\n";
  meta << "***\nAcc1 = " << dnacc1/dnits<<", DRAcc = "<<dnbounce/(dnbounce+dnrej)<<", fr succ bou ="<<dnbounce/dnits<< ", udot mean="<<udotsum/(dnbounce+dnrej)<<"\n";
  meta<< "Time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() <<"\n";
  meta.close();
  cout << "outfiles: "<<outfname<<" and "<<outmetaname<<"\n";
}

//***********************************************************
// Inputs to main (defaults in brackets)
// 1. Which target to look at (0)
// 2. Number of iterations (1)
// 3. delta (0.1)
// 4. kappa (0.1)
// 5. Print frequency (every 10 iterations)
// 6. Thinning factor (no thinning)
//
// Target 0 = 100-dimensional isotropic Gaussian - Fig 2 (left).
// Target 1 = 50-dimensional with condition number 100 - Fig 3 right.
// Target 2 = 50-dimensional, light tailed - Appendix C.1


int main(int argc, const char** pargv)  {
  int tst=0, nits=1, thin=1, prt=10;
  double kappa=0.1, delta=0.1;

  if (argc==1) {
    cout << pargv[0]<<" targno(0/1/2) [nits=1] [delta=0.1] [kappa=0.1] [prtfreq=10] [thin=1]\n";
  }
  if (argc>1) {
    tst=atoi(pargv[1]);
  }
  if (argc>2) {
    nits=atoi(pargv[2]);
  }
  if (argc>3) {
    delta=atof(pargv[3]);
  }
  if (argc>4) {
    kappa=atof(pargv[4]);
  }
  if (argc>5) {
    prt=atoi(pargv[5]);
  }
  if (argc>6) {
    thin=atoi(pargv[6]);
  }

  arma::arma_rng::set_seed_random();

  // Isotropic Gaussian from Fig 2
  if (tst==0) { 
    const int d=100;
    arma::vec x0=arma::randn(d);
    arma::vec u0=arma::normalise(arma::randn(d));
    vector<double> theta={1,1};
    arma::mat A=arma::eye(d,d);
    
    dbps(nits, x0, u0,  theta, theta, delta, kappa, A,
	 &l_null, &gl_null, &l_GaussLinSD, &gl_GaussLinSD,
	 "isod100",thin, prt);
  }
  

  // Example from Figure 3
  if (tst==1) { // Gauss in d=50 with condition number 100
    const int d=50;
    const double maxeigen=10.0;
    arma::vec sds=arma::linspace<arma::dvec>(1,maxeigen,d);
    arma::vec x0=sds%arma::randn(d);
    arma::vec u0=arma::normalise(arma::randn(d));
    vector<double> theta={1,maxeigen};
    arma::mat A=arma::eye(d,d);
      
    dbps(nits, x0, u0,  theta, theta, delta, kappa, A,
	 &l_null, &gl_null, &l_GaussLinSD, &gl_GaussLinSD,
	 "linspcd50",thin, prt);
  }

  // Light-tailed target from 4.3 and appendix 
  if (tst==2) { // d=50 with condition number 10
    const int d=50, a=4;
    const double maxeigen=10.0, gamma=10.0, da=(double)a;
    const double rstar=exp((log((double)d -1.0)/da));
    arma::dvec x0=arma::normalise(arma::randn(d));
    arma::dvec sds=arma::linspace<arma::dvec>(1,maxeigen,d);
    x0=x0%sds*rstar*gamma; // tails
    arma::vec u0=arma::normalise(arma::randn(d));
    vector<double> theta={(double)a,(double)1,(double)maxeigen};
    arma::mat A=arma::eye(d,d);
    cout << "rstar="<<rstar<<", lpri when ||x||=rstar is"<<-exp(da*log(rstar))/da<<"\n";
    
    dbps(nits, x0, u0,  theta, theta, delta, kappa, A,
	 &l_null, &gl_null, &l_PowLinSc, &gl_PowLinSc,
	 "powlinspcd50",thin, prt);

  }

  
  return 0;
}
