//Gauss-Seidel method with red-black coloring for 2D Laplace equation
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
void GS(long N, double h, double *u, double *f, double *s){
  // update red part
  #ifdef _OPENMP
    #pragma omp parallel for
  #endif
  for (long i=1; i <= N; i++){
    for (long j = 1; j <= N; j++){
      u[i+j*(N+2)] = (h*h*f[i+j*(N+2)]-u[i-1+j*(N+2)] + u[i+(j-1)*(N+2)] + u[i+1+j*(N+2)] + u[i+(j+1)*(N+2)])/4;
    }
  }
  // update black part
  #ifdef _OPENMP
    #pragma omp parallel for
  #endif
  for (long i=1; i <= N; i++){
    for (long j = 1; j <= N; j++){
      u[i+j*(N+2)] = (h*h*f[i+j*(N+2)] + s[i+j*(N+2)]) / 4;
    }
  }
}
