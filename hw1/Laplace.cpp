// Solve 1D Laplace equation: BVP,
// iterative method for Au = f
//$ g++ -std=c++11 -O3 Laplace.cpp && ./a.out -N 100 -type 0
// -type 0: Jacobi, else: Gauss-Seidel

#include <stdio.h>
#include <math.h>
#include "utils.h"

// compute l2 residual of Au-f
double Residual(long n, double *u, double *a, double *f){
  double res = 0;
  for (int i = 0; i < n; i++) {
    double Au = 0;
    for (int j = 0; j < n; j++){
      Au = Au + a[i+j*n]*u[j];
    }
    res = res + pow(Au - f[i], 2);
  }
  res = sqrt(res);
  return res;
}

// Jacobi iteration; s --  to record previous Au (j!=i)
void Jacobi(long n, double *u, double *s, double *a, double *f){
  for (int i = 0; i < n; i++){
    s[i] = 0;
    for (int j = 0; j < n; j++){
      if (j != i){
        s[i] += a[i+j*n]*u[j];
      }
    }
  }
  for (int i = 0; i < n; i++){
    u[i] = (f[i] - s[i])/a[i+i*n];
  }
}


// Gauss-Seidel iteration; s --  to record previous Au (j>i)
void Gauss_Seidel(long n, double *u, double *s, double *a, double *f){
  for (int i = 0; i < n; i++){
    s[i] = 0;
    for (int j = i+1; j < n; j++){
      s[i] += a[i+j*n]*u[j];
    }
  }
  for (int i = 0; i < n; i++){
    u[i]= f[i] - s[i];
    for (int j = 0; j < i; j++){
      u[i] -= a[i+j*n]*u[j];
    }
    u[i] = u[i]/a[i+i*n];
  }
}

void iteration(long n, double *u, double *s, double *a, double *f, int max_ite, double tol, int type){
  double res0 = Residual(n, u, a, f);
  double rel = 1;
  int ite = 0;
  if (type == 0){
    printf("Jacobi \n");
  }else{
    printf("Gauss_Seidel \n");
  }
  printf(" Iteration     Residual \n");
  while (ite < max_ite && rel > tol) {
    ite = ite + 1;
    if (type == 0){
      Jacobi(n, u, s, a, f);
    }else{
      Gauss_Seidel(n, u, s, a, f);
    }
    double res = Residual(n, u, a, f);
    rel = res/res0;
    printf("%10d %10f \n", ite, res);
  }
}

int main(int argc, char** argv) {
  long n = read_option<long>("-N", argc, argv);
  int type = read_option<int>("-type", argc, argv);
  long max_ite = read_option<long>("-ite", argc, argv, "5000");

  double* f = (double*) malloc(n * sizeof(double));
  double* a = (double*) malloc(n * n * sizeof(double));
  double* u = (double*) malloc(n * sizeof(double));
  double* s = (double*) malloc(n * sizeof(double));

  double h = 1/(n+1);

  for (long i = 0; i < n; i++) f[i] = 1.;
  for (long i = 0; i < n; i++) u[i] = 0.;
  for (long i = 0; i < n; i++) s[i] = 0.;
  for (long i = 0; i < n; i++){
    for (long j = 0; j < n; j++){
      a[i+i*n]=0;
      if(i==j){
        a[i+j*n] = 2/h/h;
      }
      if(i==j-1 || i==j+1){
        a[i+j*n] = -1/h/h;
      }
    }
  }

  const double tol = 1e-6;

  Timer t;
  t.tic();
  iteration(n, u, s, a, f, max_ite, tol, type);
  double time = t.toc();
  printf("time = %f\n", time);
  
  free(f);
  free(a);
  free(u);
  free(s);

  return 0;
}
