//Gauss-Seidel method with red-black coloring for 2D Laplace equation
// To set number of threads: setenv OMP_NUM_THREADS 8
// To use openmp: g++ -std=c++11 -fopenmp gs2D-omp.cpp -o gs2D-omp && ./gs2D-omp 1000 100
// Not use openmp: g++ -std=c++11  gs2D-omp.cpp -o gs2D && ./gs2D 1000 100

#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#else
#include "utils.h"
#endif

// Gauss-Seidel iteration update, s for storing the internal states
void GS(long N, double h, double *u, double *f){
  // update red part
  #ifdef _OPENMP
    #pragma omp parallel for
  #endif
  for (long j = 1; j <= N; j++){
    for (long i = 2 - j%2 ; i <= N; i = i + 2){
      u[i+j*(N+2)] = (h*h*f[i+j*(N+2)] + u[i-1+j*(N+2)] + u[i+(j-1)*(N+2)] + u[i+1+j*(N+2)] + u[i+(j+1)*(N+2)])/4;
    }
  }
  // update black part
  #ifdef _OPENMP
    #pragma omp parallel for
  #endif
  for (long j = 1; j <= N; j++){
    for (long i = j%2 + 1 ; i <= N; i = i + 2){
      u[i+j*(N+2)] = (h*h*f[i+j*(N+2)] + u[i-1+j*(N+2)] + u[i+(j-1)*(N+2)] + u[i+1+j*(N+2)] + u[i+(j+1)*(N+2)])/4;
    }
  }
}

// compute residual
double Residual(long N, double *u, double *f){
  double res = 0.;
  #ifdef _OPENMP
    #pragma omp parallel for reduction(+:res)
  #endif
  for (long j = 1; j <= N; j++){
    for (long i = 1; i <= N; i++){
      res += pow((4*u[i+j*(N+2)] - u[i-1+j*(N+2)] - u[i+(j-1)*(N+2)] -
       u[i+1+j*(N+2)] - u[i+(j+1)*(N+2)])*(N+1)*(N+1) - f[i+j*(N+2)], 2);
    }
  }
  res = pow(res, 0.5);
  return res;
}

// Gauss-Seidel full iterations
void GS_iter(long N, double *u, double *f, int max_ite, double tol){
  double h = 1./(N+1);
  double res0 = Residual(N, u, f);
  double rel = 1;
  double res;
  int ite = 0;
  // // uncomment this part if want to print iterations
  printf("Gauss-Seidel Iterations:\n");
  // printf(" Iteration       Residual \n");
  // printf("%10d %10f \n", ite, res0);
  while (ite < max_ite && rel > tol) {
    ite = ite + 1;
    GS(N, h, u, f);
    // res = Residual(N, u, f);
    // rel = res/res0;
    // printf("%10d %10f \n", ite, res);
  }
  res = Residual(N, u, f);
  printf("res0 = %f, res =  %f, factor = %f\n", res0, res, res0/res);
}



int main(int argc, char** argv) {
  long N, max_ite;
  sscanf(argv[1], "%d", &N);
  sscanf(argv[2], "%d", &max_ite);


  double* f = (double*) malloc((N+2)*(N+2) * sizeof(double));
  double* u = (double*) malloc((N+2)*(N+2) * sizeof(double));



  // build f and u0
  for (long i = 0; i < (N+2)*(N+2); i++) f[i] = 1.;
  for (long i = 0; i < (N+2)*(N+2); i++) u[i] = 0.;



  const double tol = 1e-6;


  #ifdef _OPENMP
    double tt = omp_get_wtime();
  #else
    Timer t;
    t.tic();
  #endif

  GS_iter(N, u, f, max_ite, tol);

  #ifdef _OPENMP
    double time = omp_get_wtime() - tt;
  #else
    double time = t.toc();
  #endif
  printf("time = %f\n", time);


  free(f);
  free(u);


  return 0;
}
