#!/bin/bash

# Set number of OpenMP thread to use
export OMP_NUM_THREADS=32

# Run the program
mpirun shifter --image=nielsbohr/hpc-notebook ./ct_parallel --num-voxels 256 --input ~/slurm_readonly/ct_data

