#!/bin/bash
#SBATCH --exclusive

# set loop scheduling to static
export OMP_SCHEDULE=static
# Schedule one thread per core. Change to "threads" for hyperthreading
export OMP_PLACES=cores
# Place threads as close to each other as possible
export OMP_PROC_BIND=close

# Set number of OpenMP thread to use (this should be 64 cores / number of ranks pr node)
export OMP_NUM_THREADS=32

# Run the program
mpirun singularity exec ~/modi_images/hpc-notebook_latest.sif ./ct_parallel --num-voxels 256 --input ct_data

