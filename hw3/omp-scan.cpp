#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}


void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  if (n == 0) return;
  prefix_sum[0] = 0;

  int nthreads;
  long chunk_size;

  int tid;


  #pragma omp parallel private(tid)
  {
    #pragma omp master
    {
      nthreads = omp_get_num_threads();
      printf("Number of threads = %d\n", nthreads);
      chunk_size = (n-1)/nthreads;
      if ((n-1)%nthreads > 0){
        chunk_size += 1;
      }
    }

    #pragma omp barrier

    #pragma omp for schedule(static, chunk_size)
    for (long i = 1; i < n; i++) {
      prefix_sum[i] = prefix_sum[i-1] + A[i-1];
      tid = omp_get_thread_num();
      // printf("tid = %d, i=%d\n", tid, i);
    } // parallel for each chunk
  }

    // printf("chunk_size = %d\n", chunk_size);
    long* cor_vec = (long*) malloc(nthreads * sizeof(long));
    cor_vec[0] = 0;
    for (long j=1; j < nthreads; j++){
      cor_vec[j] = cor_vec[j-1]+prefix_sum[j*chunk_size];
    }// store correction

    // #pragma omp barrier

    #pragma omp parallel for
    for (long i = 1+chunk_size; i < n; i++){
      prefix_sum[i] += cor_vec[(i-1)/chunk_size];
    } // parallel for adding correction term


  free(cor_vec);

}

int main() {
  long N = 100000000;
  // long N = 12;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();
  for (long i = 0; i < N; i++) B1[i] = 0;


  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);


  // for (long i = 0; i < N; i++) printf("i = %d, B0= %ld, B1=%ld\n",i, B0[i], B1[i]);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
