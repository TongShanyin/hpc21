// Solve 1D Laplace equation: BVP, use sparsity of A
// iterative method for Au = f
//$ g++ -std=c++11 -O3 Laplace2.cpp && ./a.out -N 100 -type 0
// -type 0: Jacobi, else: Gauss-Seidel

#include <stdio.h>
#include <math.h>
#include "utils.h"

// compute l2 residual of Au-f
double Residual(long n, double *u){
  double res = 0;
  double Au = 0;
  for (int i = 0; i < n; i++) {
    Au = 0;
    if(i-1>=0){
      Au = Au - u[i-1];
    }
    if(i+1<n){
      Au = Au - u[i+1];
    }
    Au = Au + 2*u[i];
    Au = Au*(n+1)*(n+1);
    res = res + pow(Au - 1, 2);
  }
  res = pow(res, 0.5);
  return res;
}

// Jacobi iteration; s --  to record previous Au (j!=i)
void Jacobi(long n, double *u, double *s){
  for (int i = 0; i < n; i++){
    s[i] = 0;
    if(i-1>=0){
      s[i] += u[i-1];
    }
    if(i+1<n){
      s[i] += u[i+1];
    }
    s[i]=s[i]*(n+1)*(n+1);
  }
  for (int i = 0; i < n; i++){
    u[i] = (1. - s[i])/2/(n+1)/(n+1);
  }
}


// Gauss-Seidel iteration; s --  to record previous Au (j>i)
void Gauss_Seidel(long n, double *u, double *s){
  for (int i = 0; i < n; i++){
    s[i] = 0;
    if(i+1<n){
      s[i] += u[i+1];
    }
    s[i]=s[i]*(n+1)*(n+1);
  }
  for (int i = 0; i < n; i++){
    u[i]= 1. - s[i];
    if(i-1>=0){
      u[i] += u[i-1]*(n+1)*(n+1);
    }
    u[i] = u[i]/2/(n+1)/(n+1);
  }
}

void iteration(long n, double *u, double *s, int max_ite, double tol, int type){
  double res0 = Residual(n, u);
  double rel = 1;
  int ite = 0;
  if (type == 0){
    printf("Jacobi \n");
  }else{
    printf("Gauss_Seidel \n");
  }
  printf(" Iteration       Residual \n");
  printf("%10d %10f \n", ite, res0);
  while (ite < max_ite && rel > tol) {
    ite = ite + 1;
    if (type == 0){
      Jacobi(n, u, s);
    }else{
      Gauss_Seidel(n, u, s);
    }
    double res = Residual(n, u);
    rel = res/res0;
    printf("%10d %10f \n", ite, res);
  }
  printf("factor=%f\n", res0/res);
}

int main(int argc, char** argv) {
  long n = read_option<long>("-N", argc, argv);
  int type = read_option<int>("-type", argc, argv);
  long max_ite = read_option<long>("-ite", argc, argv, "5000");

  double* u = (double*) malloc(n * sizeof(double));
  double* s = (double*) malloc(n * sizeof(double));


  // initialize u0 and s
  for (long i = 0; i < n; i++) u[i] = 0.;
  for (long i = 0; i < n; i++) s[i] = 0.;


  const double tol = 1e-6;


  Timer t;
  t.tic();
  iteration(n, u, s, max_ite, tol, type);
  double time = t.toc();
  printf("time = %f\n", time);


  free(u);
  free(s);

  return 0;
}
