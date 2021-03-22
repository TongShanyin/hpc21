/******************************************************************************
* FILE: omp_bug2.c
* DESCRIPTION:
*   Another OpenMP program with a bug.
* AUTHOR: Blaise Barney
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[])
{
int nthreads, i, tid;
float total;

/*** Spawn parallel region ***/
// need to specify tid as a private variable, need to use reduction since we do sum.
#pragma omp parallel private(tid) reduction(+:total)
  {
  /* Obtain thread number */
  tid = omp_get_thread_num();
  /* Only master thread does this */
  if (tid == 0) {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d is starting...\n",tid);

  #pragma omp barrier

  /* do some work */
  total = 0.0;
  #pragma omp for schedule(dynamic,10)
  for (i=0; i<1000000; i++)
     total = total + i*1.0;

  // printf ("Thread %d is done! Total= %e\n",tid,total);
    printf ("Thread %d is done!\n",tid);
  } /*** End of parallel region ***/
  printf ("Total= %e\n",total); // print total when parallel ends
  // the total value is different from the analytic solution, but I try using one thread, still not the same as the analytic one,
  // might from the round-off error.
}
