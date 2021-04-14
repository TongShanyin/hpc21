#!/bin/bash

#SBATCH --job-name=jacobi
#SBATCH --output=jacobi.out
#SBATCH --error=jacobi.err
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB 
#SBATCH --time=00:10:00 
#SBATCH --verbose

module purge 
module load  amber/openmpi/intel/20.06 

export OMP_NUM_THREADS=4
mpiexec ./jacobi2D-mpi 1024 100
