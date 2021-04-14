#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <mpi.h>
#include <string.h>

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *lu, int lN, double invhsq) {
  int i,j;
  double tmp, gres = 0.0, lres = 0.0;

#pragma omp parallel for default(none) shared(lu,lN,invhsq) private(i,j,tmp) reduction(+:lres)
  for (j = 1; j <= lN; j++){
    for (i = 1; i <= lN; i++){
      tmp = ((4.0*lu[i+j*(lN+2)] - lu[i-1+j*(lN+2)] - lu[i+1+j*(lN+2)] - lu[i+(j-1)*(lN+2)] - lu[i+(j+1)*(lN+2)]) * invhsq - 1);
      lres += tmp * tmp;
    }
  }
  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}


int main(int argc, char * argv[]) {
  int mpirank, i,j, p, N, lN, iter, max_iters;
  MPI_Status status, status1, status2, status3;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

  sscanf(argv[1], "%d", &N);
  sscanf(argv[2], "%d", &max_iters);
# pragma omp parallel
  {
#ifdef _OPENMP
    int my_threadnum = omp_get_thread_num();
    int numthreads = omp_get_num_threads();
#else
    int my_threadnum = 0;
    int numthreads = 1;
#endif
    printf("Hello, I'm thread %d out of %d on mpirank %d\n", my_threadnum, numthreads, mpirank);
  }
  /* compute number of unknowns handled by each process */
  int lp = pow(p,0.5); // process per row/column
  lN = N / lp;
  if ((N*N - p*lN*lN != 0) && mpirank == 0 ) {
    printf("N: %d, local N: %d\n", N, lN);
    printf("Exiting. N must be a multiple of p\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  /* Allocation of vectors, including left/upper and right/lower ghost points */
  double * lu    = (double *) calloc(sizeof(double), (lN + 2)*(lN+2));
  double * lunew = (double *) calloc(sizeof(double), (lN + 2)*(lN+2));
  double * lutemp;

  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-5;

  /* initial residual */
  gres0 = compute_residual(lu, lN, invhsq);
  gres = gres0;

  for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {

#pragma omp parallel for default(none) shared(lN,lunew,lu,hsq) private(i,j)
    /* Jacobi step for local points */
    for (j = 1; j <= lN; j++){	  
      for (i = 1; i <= lN; i++){
        lunew[i+j*(lN+2)]  = 0.25 * (hsq + lu[i-1+j*(lN+2)] + lu[i+1+j*(lN+2)] + lu[i+(j-1)*(lN+2)] + lu[i+(j+1)*(lN+2)]);
      }
    }
    /* communicate ghost values */
    for (i = 1; i <= lN; i++){
      if (mpirank< p-lp) {
        /* If not the right process, send/recv bdry values to the right */
      	MPI_Send(&(lunew[i+lN*(lN+2)]), 1, MPI_DOUBLE, mpirank+lp, 124, MPI_COMM_WORLD);
      	MPI_Recv(&(lunew[i+(lN+1)*(lN+2)]), 1, MPI_DOUBLE, mpirank+lp, 123, MPI_COMM_WORLD, &status);
      }
      if (mpirank> lp-1) {
        /* If not the left process, send/recv bdry values to the left */
        MPI_Send(&(lunew[i+1*(lN+2)]), 1, MPI_DOUBLE, mpirank-lp, 123, MPI_COMM_WORLD);
        MPI_Recv(&(lunew[i+0*(lN+2)]), 1, MPI_DOUBLE, mpirank-lp, 124, MPI_COMM_WORLD, &status1);
      }
    }

    for (j = 1; j <= lN; j++){
      if (mpirank%lp != lp-1 ) {
        /* If not the bottom process, send/recv bdry values to the bottom */
        MPI_Send(&(lunew[lN+j*(lN+2)]), 1, MPI_DOUBLE, mpirank+1, 126, MPI_COMM_WORLD);
        MPI_Recv(&(lunew[lN+1+j*(lN+2)]), 1, MPI_DOUBLE, mpirank+1, 125, MPI_COMM_WORLD, &status2);
      }
      if (mpirank%lp != 0) {
        /* If not the top process, send/recv bdry values to the top */
        MPI_Send(&(lunew[1+j*(lN+2)]), 1, MPI_DOUBLE, mpirank-1, 125, MPI_COMM_WORLD);
        MPI_Recv(&(lunew[0+j*(lN+2)]), 1, MPI_DOUBLE, mpirank-1, 126, MPI_COMM_WORLD, &status3);
      }
    }

    /* copy newu to u using pointer flipping */
    lutemp = lu; lu = lunew; lunew = lutemp;
    if (0 == (iter % 10)) {
      gres = compute_residual(lu, lN, invhsq);
      if (0 == mpirank) {
	printf("Iter %d: Residual: %g\n", iter, gres);
      }
    }
  }

  /* Clean up */
  free(lu);
  free(lunew);

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == mpirank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();
  return 0;
}
