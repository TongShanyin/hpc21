#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <string.h>
#include <iostream>
//#include<stdlib.h>

double time_ring(long Nrepeat, long Nsize, MPI_Comm comm) {
  int rank, world_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &world_size);

  int* msg = (int*) malloc(Nsize*sizeof(int));
  for (long i = 0; i < Nsize; i++) msg[i] = 0;

  MPI_Barrier(comm);
  double tt = MPI_Wtime();

  for (long repeat  = 0; repeat < Nrepeat; repeat++) {
    // loop
    for (int proc = 0; proc < world_size-1; proc++){
      MPI_Status status;
      if (rank == proc){
	for (long i = 0; i < Nsize; i++) msg[i] += rank;
        MPI_Send(msg, Nsize, MPI_INT, proc+1, repeat, comm);
      }
      if (rank == proc+1){
      	MPI_Recv(msg, Nsize, MPI_INT, proc, repeat, comm, &status);
      }
    }
    MPI_Status status;
    if (rank == world_size-1){
      for (long i = 0; i < Nsize; i++) msg[i] += rank;
      MPI_Send(msg, Nsize, MPI_INT, 0, repeat, comm);
    }
    if (rank == 0){
      MPI_Recv(msg, Nsize, MPI_INT, world_size-1, repeat, comm, &status);
    }
  }

//  MPI_Barrier(comm);
//  printf("Rank:%d message is %d\n",rank, msg[0]);
  
  
  tt = MPI_Wtime() - tt;

  free(msg);
  return tt;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank,world_size;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &world_size);

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", rank, world_size, processor_name);

  long Nrepeat = 10000;
  double tt = time_ring(Nrepeat, 1, comm);
  if (!rank) printf("ring latency: %e ms\n", tt/Nrepeat/world_size * 1000);

  Nrepeat = 10000;
  long Nsize = 1000000;
  tt = time_ring(Nrepeat, Nsize, comm);
  if (!rank) printf("ring bandwidth: %e GB/s\n", (Nsize*Nrepeat*world_size*sizeof(int))/tt/1e9);

  MPI_Finalize();
}

