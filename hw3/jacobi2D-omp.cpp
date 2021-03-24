// Jacobi method for Laplace equation in 2D

#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#else
#include "utils.h"
#endif

// Jacobi iteration update, s for storing the internal states
void Jacobi(long N, double h, double *u, double *f, double *s){
  // double h = 1/(N+1);
  #ifdef _OPENMP
    #pragma omp parallel for
  #endif
  for (long i=1; i <= N; i++){
    for (long j = 1; j <= N; j++){
      s[i+j*(N+2)] = u[i-1+j*(N+2)] + u[i+(j-1)*(N+2)] + u[i+1+j*(N+2)] + u[i+(j+1)*(N+2)];
    }
  }
  #ifdef _OPENMP
    #pragma omp parallel for
  #endif
  for (long i=1; i <= N; i++){
    for (long j = 1; j <= N; j++){
      u[i+j*(N+2)] = (h*h*f[i+j*(N+2)] + s[i+j*(N+2)]) / 4;
    }
  }
  // #ifdef _OPENMP
  //   #pragma omp parallel for
  // #endif
  // for (long i=1; i <= N; i++){
  //   for (long j = 1; j <= N; j++){
  //     u[i+j*(N+2)] = (h*h*f[i+j*(N+2)] + u[i-1+j*(N+2)] + u[i+(j-1)*(N+2)] +
  //      u[i+1+j*(N+2)] + u[i+(j+1)*(N+2)]) / 4;
  //   }
  // }
}

// compute residual
double Residual(long N, double *u, double *f){
  double res = 0;
  #ifdef _OPENMP
    #pragma omp parallel for reduction(+:res)
  #endif
  for (long i=1; i <= N; i++){
    for (long j = 1; j <= N; j++){
      res += pow((4*u[i+j*(N+2)] - u[i-1+j*(N+2)] - u[i+(j-1)*(N+2)] -
      u[i+1+j*(N+2)] - u[i+(j+1)*(N+2)])*(N+1)*(N+1) - f[i+j*(N+2)], 2);
    }
  }
  res = pow(res, 0.5);
  return res;
}

// Jacobi full iterations
void Jacobi_iter(long N, double *u, double *f, double *s, int max_ite, double tol){
  double h = 1/(N+1);
  double res0 = Residual(N, u, f);
  double rel = 1;
  int ite = 0;
  printf("Jacobi Iterations:\n");
  printf(" Iteration       Residual \n");
  printf("%10d %10f \n", ite, res0);
  while (ite < max_ite && rel > tol) {
    ite = ite + 1;
    Jacobi(N, h, u, f);
    double res = Residual(N, u, f, s);
    rel = res/res0;
    printf("%10d %10f \n", ite, res);
  }
}

int main(int argc, char** argv) {
  long N = read_option<long>("-N", argc, argv);
  long max_ite = read_option<long>("-ite", argc, argv, "5000");

  double* f = (double*) malloc((N+2)*(N+2) * sizeof(double));
  double* u = (double*) malloc((N+2)*(N+2) * sizeof(double));
  double* s = (double*) malloc((N+2)*(N+2) * sizeof(double));



  // build f and u0
  for (long i = 0; i < (N+2)*(N+2); i++) f[i] = 1.;
  for (long i = 0; i < (N+2)*(N+2); i++) u[i] = 0.;
  for (long i = 0; i < (N+2)*(N+2); i++) s[i] = 0.;


  const double tol = 1e-6;
  double res0 = Residual(N, u, f);

  #ifdef _OPENMP
    tt = omp_get_wtime();
  #else
    Timer t;
    t.tic();
  #endif
  Jacobi_iter(N, u, f, s, max_ite, tol);
  #ifdef _OPENMP
    double time = omp_get_wtime() - tt;
  #else
    double time = t.toc();
  #endif
  printf("time = %f\n", time);

  double res = Residual(N, u, f);
  printf("factor = %f\n", res0/res);

  free(f);
  free(u);
  free(s);

  return 0;
}
