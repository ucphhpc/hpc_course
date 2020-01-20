#!/bin/bash

# Set number of OpenMP thread to use
export OMP_NUM_THREADS=32

# Run the program
mpirun singularity exec ~/modi_images/hpc-notebook-latest.simg ./ct_parallel --num-voxels 256 --input ct_data

