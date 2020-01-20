#!/bin/bash

# Set number of OpenMP thread to use
export OMP_NUM_THREADS=32

# Run the program
mpirun singularity ~/modi_images/hpc-notebook-latest.simg ./ct_parallel --num-voxels 256 --input ~/modi_mount/module5/CT_Reconstruction/ct_data

