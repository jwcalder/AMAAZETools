/* svi_computations.h - 
 *
 *  Subroutines specific to computing the spherical volume invariant
 *
 *  Author: Jeff Calder, 2018.
 */
 

void svipca(double *P_ptr, int n, int *T_ptr, int m, bool *ID, double r, double eps_svi, double eps_pca, bool prog, double *S, double *M_ptr);
void svi(double *P_ptr, int n, int *T_ptr, int m, bool *ID, double r, double eps, bool prog, double *S, double *Q);
bool svi_error_computation(double lmax, double eta, double r, double r2, double eps);
double svi_integrate_approx(double *x, double *y, double *z, double A, double eta, double r2, double r3, int i, int j);
double svi_integrate_exact(double *x, double *y, double *z, double r2, double r3, int i, int j);
double svigamma(int **T, double **P, int **L, int i);
bool pcami_error_computation(double lmax, double eta, double r, double r2, double eps);
bool pcacij_error_computation(double lmax, double eta, double r, double r2, double eps);
double pcami_integrate_exact(double *x, double *y, double *z, double r2, double r3, int i, int j);
double pcacij_integrate_exact(double *x, double *y, double *z, double r2, double r3, int i, int j);
double pcami_integrate_approx(double *x, double *y, double *z, double A, double eta, double r2, double r3, int i, int j);
double pcacij_integrate_approx(double *x, double *y, double *z, double A, double eta, double r2, double r3, int i, int j);
