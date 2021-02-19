// Null - returns zero
double l_null(const arma::vec &x, const std::vector<double> &thetas);
arma::dvec gl_null(const arma::vec &x, const std::vector<double> &thetas);
double gldote_null(const arma::vec &x, const arma::vec &e, const std::vector<double> &thetas);
// Gaussian ISO
double l_GaussIso(const arma::vec &x, const std::vector<double> &thetas);
arma::dvec gl_GaussIso(const arma::vec &x, const std::vector<double> &thetas);
double gldote_GaussIso(const arma::vec &x, const arma::vec &e, const std::vector<double> &thetas);
// Powered exponential ISO
double l_PowIso(const arma::vec &x, const std::vector<double> &thetas);
arma::dvec gl_PowIso(const arma::vec &x, const std::vector<double> &thetas);
double gldote_PowIso(const arma::vec &x, const arma::vec &e, const std::vector<double> &thetas);
// Gaussian ANISO
double l_GaussDiag(const arma::vec &x, const std::vector<double> &theta);
arma::dvec gl_GaussDiag(const arma::vec &x, const std::vector<double> &theta);
double l_GaussLinSD(const arma::vec &x, const std::vector<double> &thetas);
arma::dvec gl_GaussLinSD(const arma::vec &x, const std::vector<double> &thetas);
double gldote_GaussLinSD(const arma::vec &x, const arma::vec &e, const std::vector<double> &thetas);
double l_GaussLinH(const arma::vec &x, const std::vector<double> &thetas);
arma::dvec gl_GaussLinH(const arma::vec &x, const std::vector<double> &thetas);
double gldote_GaussLinH(const arma::vec &x, const arma::vec &e, const std::vector<double> &thetas);
// Powered exponential ANISO
double l_PowLinSc(const arma::vec &x, const std::vector<double> &thetas);
arma::dvec gl_PowLinSc(const arma::vec &x, const std::vector<double> &thetas);
double gldote_PowLinSc(const arma::vec &x, const arma::vec &e, const std::vector<double> &thetas);


// Student-t; theta=[nu,mu,SigInv]
double l_Student_t_gen(const arma::vec &x, const std::vector<double> &thetas);

// Bimodal with means at +/- theta[0] and prob(+theta[0])=theta[1]
double l_GaussBimod(const arma::vec &x, const std::vector<double> &thetas);
